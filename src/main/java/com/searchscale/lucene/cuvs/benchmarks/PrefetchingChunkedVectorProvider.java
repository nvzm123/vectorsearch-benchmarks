package com.searchscale.lucene.cuvs.benchmarks;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Double-buffered, prefetching variant of {@link ChunkedVectorProvider} for uncompressed {@code
 * .fbin}/{@code .fvecs} files. A background reader thread fills the next chunk while the ingest
 * thread consumes the current one, so the sequential disk read overlaps with the per-vector consume
 * (addDocument) instead of serializing behind it. Combined with {@link #get(int, float[])} it also
 * unpacks directly from the chunk buffer into a caller-reused array, avoiding a per-vector
 * allocation.
 *
 * <p><b>Forward-only, single-consumer.</b> {@link #get} must be called with monotonically
 * non-decreasing indices from a single thread (i.e. {@code numIndexThreads=1}). Random or concurrent
 * access throws — use {@link ChunkedVectorProvider} for those.
 */
public class PrefetchingChunkedVectorProvider implements VectorProvider {

  private static final Logger log =
      LoggerFactory.getLogger(PrefetchingChunkedVectorProvider.class.getName());
  private static final boolean USE_BULK_ROW_COPY =
      !Boolean.getBoolean("cuvs.benchmark.scalarVectorRowCopy");

  private enum Format {
    FBIN,
    FVECS
  }

  private static final class Chunk {
    final ByteBuffer buf;
    final FloatBuffer floats;
    final long start;
    final int len;

    Chunk(ByteBuffer buf, long start, int len) {
      this.buf = buf;
      if (buf == null) {
        this.floats = null;
      } else {
        ByteBuffer filled = buf.duplicate().order(ByteOrder.LITTLE_ENDIAN);
        filled.flip();
        this.floats = filled.asFloatBuffer();
      }
      this.start = start;
      this.len = len;
    }
  }

  /** Sentinel placed on the ready queue once the reader has produced the final chunk. */
  private static final Chunk POISON = new Chunk(null, -1, 0);

  private final FileChannel channel;
  private final Format format;
  private final int dimension;
  private final int vectorCount;
  private final int vectorSize;
  private final long headerBytes;
  private final int prefixBytes;
  private final int chunkVectors;
  private final long firstVector; // absolute index of the first vector this reader serves

  private final BlockingQueue<ByteBuffer> free = new ArrayBlockingQueue<>(2);
  private final BlockingQueue<Chunk> ready = new ArrayBlockingQueue<>(2);
  private final Thread reader;
  private volatile IOException readerError;

  private Chunk current; // consumer-owned; the chunk currently being served

  public static boolean supports(String filePath) {
    return ChunkedVectorProvider.supports(filePath);
  }

  /** Reads the whole file (or the first {@code maxVectors}) from index 0. */
  public PrefetchingChunkedVectorProvider(String filePath, int maxVectors, int chunkSizeMB)
      throws IOException {
    this(filePath, 0, maxVectors, chunkSizeMB);
  }

  /**
   * Reads the contiguous range {@code [firstVector, firstVector + count)} (or to end of file if
   * {@code count <= 0}) so each thread of the parallel partitioned build streams only its own slice.
   * {@link #get} then serves absolute file indices within that range.
   */
  public PrefetchingChunkedVectorProvider(
      String filePath, int firstVector, int count, int chunkSizeMB) throws IOException {
    if (filePath.contains("fbin")) {
      this.format = Format.FBIN;
      this.headerBytes = 8;
      this.prefixBytes = 0;
    } else if (filePath.contains("fvecs")) {
      this.format = Format.FVECS;
      this.headerBytes = 0;
      this.prefixBytes = 4;
    } else {
      throw new IllegalArgumentException(
          "PrefetchingChunkedVectorProvider supports only .fbin/.fvecs");
    }

    this.channel = FileChannel.open(Path.of(filePath), StandardOpenOption.READ);

    int dim;
    int end;
    if (format == Format.FBIN) {
      // header: [num_vectors int32][dimension int32]
      ByteBuffer hdr = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN);
      readFully(hdr, 0);
      hdr.flip();
      int numVectors = hdr.getInt();
      dim = hdr.getInt();
      this.vectorSize = 4 * dim;
      end = count > 0 ? (int) Math.min((long) firstVector + count, numVectors) : numVectors;
    } else {
      // fvecs: the first 4 bytes are the first vector's dimension prefix
      ByteBuffer hdr = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN);
      readFully(hdr, 0);
      hdr.flip();
      dim = hdr.getInt();
      this.vectorSize = 4 + 4 * dim;
      long total = channel.size() / vectorSize;
      end = count > 0 ? (int) Math.min((long) firstVector + count, total) : (int) total;
    }
    this.dimension = dim;
    this.firstVector = firstVector;
    this.vectorCount = end;

    // vectors per chunk, bounded so chunkVectors * vectorSize stays within an int
    long chunkBytes = (long) Math.max(1, chunkSizeMB) * 1024 * 1024;
    int maxByCap = (Integer.MAX_VALUE - 16) / vectorSize;
    this.chunkVectors = (int) Math.max(1, Math.min(chunkBytes / vectorSize, maxByCap));

    // Two reusable direct buffers: the reader fills one while the consumer drains the other.
    for (int i = 0; i < 2; i++) {
      free.add(ByteBuffer.allocateDirect(chunkVectors * vectorSize).order(ByteOrder.LITTLE_ENDIAN));
    }

    log.info(
        "PrefetchingChunkedVectorProvider: {} vectors, {} dims, format {}, {} vectors/chunk (~{}"
            + " MB), double-buffered prefetch",
        vectorCount - firstVector,
        dimension,
        format,
        chunkVectors,
        (long) chunkVectors * vectorSize / (1024 * 1024));

    this.reader = new Thread(this::readLoop, "chunk-prefetch-reader");
    this.reader.setDaemon(true);
    this.reader.start();
  }

  /** Reader thread: fill chunks front-to-back, blocking on a free buffer between chunks. */
  private void readLoop() {
    long next = firstVector;
    try {
      while (next < vectorCount) {
        ByteBuffer buf = free.take();
        int toRead = (int) Math.min(chunkVectors, vectorCount - next);
        buf.clear();
        buf.limit(toRead * vectorSize);
        readFully(buf, headerBytes + next * (long) vectorSize);
        ready.put(new Chunk(buf, next, toRead));
        next += toRead;
      }
      ready.put(POISON);
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt(); // close() requested; stop quietly
    } catch (IOException e) {
      readerError = e;
      try {
        ready.put(POISON); // unblock the consumer so it can observe the error
      } catch (InterruptedException ignored) {
        Thread.currentThread().interrupt();
      }
    }
  }

  private void advance() throws IOException {
    if (current != null) {
      try {
        free.put(current.buf); // hand the drained buffer back to the reader
      } catch (InterruptedException e) {
        Thread.currentThread().interrupt();
        throw new IOException("Interrupted returning chunk buffer", e);
      }
      current = null;
    }
    Chunk next;
    try {
      next = ready.take();
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
      throw new IOException("Interrupted awaiting next chunk", e);
    }
    if (next == POISON) {
      if (readerError != null) {
        throw new IOException("Prefetch reader failed", readerError);
      }
      throw new IOException("No more chunks available (unexpected EOF in prefetch)");
    }
    current = next;
  }

  @Override
  public float[] get(int index) throws IOException {
    float[] dst = new float[dimension];
    get(index, dst);
    return dst;
  }

  @Override
  public void get(int index, float[] dst) throws IOException {
    if (index < firstVector || index >= vectorCount) {
      throw new IndexOutOfBoundsException(
          "Index " + index + " out of bounds [" + firstVector + ", " + vectorCount + ")");
    }
    while (current == null || index >= current.start + current.len) {
      advance();
    }
    if (index < current.start) {
      throw new UnsupportedOperationException(
          "PrefetchingChunkedVectorProvider requires forward-only sequential access; got index "
              + index
              + " before current chunk start "
              + current.start);
    }
    int base = (int) (index - current.start) * vectorSize + prefixBytes;
    if (!USE_BULK_ROW_COPY || dst.length < dimension) {
      // Preserve the existing partial-copy behavior and exception type for an invalid destination.
      for (int i = 0; i < dimension; i++) {
        dst[i] = current.buf.getFloat(base + i * Float.BYTES);
      }
      return;
    }
    current.floats.get(base / Float.BYTES, dst, 0, dimension);
  }

  /** Reads until {@code buf} is full, starting at file {@code position} (absolute, positional). */
  private void readFully(ByteBuffer buf, long position) throws IOException {
    long pos = position;
    while (buf.hasRemaining()) {
      int n = channel.read(buf, pos);
      if (n < 0) {
        throw new IOException("Unexpected EOF reading at position " + pos);
      }
      pos += n;
    }
  }

  @Override
  public int size() {
    return vectorCount - (int) firstVector;
  }

  @Override
  public void close() throws IOException {
    reader.interrupt();
    channel.close();
  }
}
