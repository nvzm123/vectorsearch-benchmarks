package com.searchscale.lucene.cuvs.benchmarks;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Sequential, chunked vector provider for uncompressed {@code .fbin} / {@code .fvecs} files.
 *
 * <p>Opens the file <b>once</b> (a single persistent {@link FileChannel}) and serves {@link
 * #get(int)} from a reusable chunk buffer. The buffer is refilled with one large sequential read
 * whenever the requested index falls outside the current chunk. For the sequential access pattern
 * of single-threaded ingestion this reads the whole file front-to-back in a handful of bulk reads
 * (near-max disk throughput) while holding at most one chunk in memory.
 *
 * <p>This avoids the per-vector {@code open}/{@code seek}/{@code read}/{@code close} of {@link
 * StreamingVectorProvider} (the "fd churn"), and avoids the full in-heap materialization of {@link
 * MemoryVectorProvider}. Chunk size is configurable via {@code ingestChunkSizeMB}.
 */
public class ChunkedVectorProvider implements VectorProvider {

  private static final Logger log = LoggerFactory.getLogger(ChunkedVectorProvider.class.getName());

  private enum Format {
    FBIN,
    FVECS
  }

  private final FileChannel channel;
  private final Format format;
  private final int dimension;
  private final int vectorCount;
  private final int vectorSize; // bytes per stored vector (incl. per-vector dim prefix for fvecs)
  private final long headerBytes; // file header before the first vector (8 for fbin, 0 for fvecs)
  private final int prefixBytes; // per-vector prefix skipped before the floats (0 fbin, 4 fvecs)
  private final int chunkVectors; // vectors held per chunk
  private final ByteBuffer chunkBuffer;

  private long chunkStart = -1; // index of the first vector currently buffered
  private int chunkLen = 0; // number of vectors currently buffered

  public static boolean supports(String filePath) {
    return !filePath.endsWith(".gz") && (filePath.contains("fbin") || filePath.contains("fvecs"));
  }

  public ChunkedVectorProvider(String filePath, int maxVectors, int chunkSizeMB)
      throws IOException {
    if (filePath.endsWith(".gz")) {
      throw new IllegalArgumentException("ChunkedVectorProvider does not support compressed files");
    }
    if (filePath.contains("fbin")) {
      this.format = Format.FBIN;
      this.headerBytes = 8;
      this.prefixBytes = 0;
    } else if (filePath.contains("fvecs")) {
      this.format = Format.FVECS;
      this.headerBytes = 0;
      this.prefixBytes = 4;
    } else {
      throw new IllegalArgumentException("ChunkedVectorProvider supports only .fbin/.fvecs");
    }

    this.channel = FileChannel.open(Path.of(filePath), StandardOpenOption.READ);

    int dim;
    int count;
    if (format == Format.FBIN) {
      // header: [num_vectors int32][dimension int32]
      ByteBuffer hdr = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN);
      readFully(hdr, 0);
      hdr.flip();
      int numVectors = hdr.getInt();
      dim = hdr.getInt();
      this.vectorSize = 4 * dim;
      count = maxVectors > 0 ? Math.min(maxVectors, numVectors) : numVectors;
    } else {
      // fvecs: the first 4 bytes are the first vector's dimension prefix
      ByteBuffer hdr = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN);
      readFully(hdr, 0);
      hdr.flip();
      dim = hdr.getInt();
      this.vectorSize = 4 + 4 * dim;
      long total = channel.size() / vectorSize;
      count = maxVectors > 0 ? (int) Math.min(maxVectors, total) : (int) total;
    }
    this.dimension = dim;
    this.vectorCount = count;

    // vectors per chunk, bounded so chunkVectors * vectorSize stays within an int
    long chunkBytes = (long) Math.max(1, chunkSizeMB) * 1024 * 1024;
    int maxByCap = (Integer.MAX_VALUE - 16) / vectorSize;
    this.chunkVectors = (int) Math.max(1, Math.min(chunkBytes / vectorSize, maxByCap));
    this.chunkBuffer =
        ByteBuffer.allocateDirect(chunkVectors * vectorSize).order(ByteOrder.LITTLE_ENDIAN);

    log.info(
        "ChunkedVectorProvider: {} vectors, {} dims, format {}, {} vectors/chunk (~{} MB)",
        vectorCount,
        dimension,
        format,
        chunkVectors,
        (long) chunkVectors * vectorSize / (1024 * 1024));
  }

  @Override
  public synchronized float[] get(int index) throws IOException {
    if (index < 0 || index >= vectorCount) {
      throw new IndexOutOfBoundsException(
          "Index " + index + " out of bounds [0, " + vectorCount + ")");
    }
    if (chunkStart < 0 || index < chunkStart || index >= chunkStart + chunkLen) {
      loadChunkContaining(index);
    }
    int base = (int) (index - chunkStart) * vectorSize + prefixBytes;
    float[] vector = new float[dimension];
    for (int i = 0; i < dimension; i++) {
      vector[i] = chunkBuffer.getFloat(base + i * 4);
    }
    return vector;
  }

  private void loadChunkContaining(int index) throws IOException {
    long start = (index / chunkVectors) * (long) chunkVectors;
    int toRead = (int) Math.min(chunkVectors, vectorCount - start);
    long offset = headerBytes + start * (long) vectorSize;
    chunkBuffer.clear();
    chunkBuffer.limit(toRead * vectorSize);
    readFully(chunkBuffer, offset);
    chunkStart = start;
    chunkLen = toRead;
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
    return vectorCount;
  }

  @Override
  public void close() throws IOException {
    channel.close();
  }
}
