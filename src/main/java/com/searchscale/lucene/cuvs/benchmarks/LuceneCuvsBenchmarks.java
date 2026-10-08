package com.searchscale.lucene.cuvs.benchmarks;

import com.nvidia.cuvs.CagraIndexParams.CagraGraphBuildAlgo;
import com.nvidia.cuvs.CagraIndexParams.CudaDataType;
import com.nvidia.cuvs.CagraIndexParams.CuvsDistanceType;
import com.nvidia.cuvs.CuVSIvfPqIndexParams;
import com.nvidia.cuvs.CuVSIvfPqParams;
import com.nvidia.cuvs.CuVSIvfPqSearchParams;
import com.nvidia.cuvs.lucene.AcceleratedHNSWParams;
import com.nvidia.cuvs.lucene.CuVS2510GPUSearchCodec;
import com.nvidia.cuvs.lucene.GPUKnnFloatVectorQuery;
import com.nvidia.cuvs.lucene.GPUSearchParams;
import com.nvidia.cuvs.lucene.Lucene101AcceleratedHNSWCodec;
import com.nvidia.cuvs.lucene.LuceneAcceleratedHNSWBinaryQuantizedCodec;
import com.nvidia.cuvs.lucene.LuceneAcceleratedHNSWScalarQuantizedCodec;
import com.nvidia.cuvs.spi.CuVSProvider;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.nio.charset.Charset;
import java.nio.file.FileVisitOption;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.Semaphore;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import org.apache.commons.io.FileUtils;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.codecs.lucene104.Lucene104Codec;
import org.apache.lucene.codecs.lucene104.Lucene104Codec.Mode;
import org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.TieredMergePolicy;
import org.apache.lucene.misc.store.HardlinkCopyDirectoryWrapper;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TopScoreDocCollectorManager;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.MMapDirectory;
import org.apache.lucene.util.PrintStreamInfoStream;
import org.mapdb.DB;
import org.mapdb.DBMaker;
import org.mapdb.IndexTreeList;
import org.mapdb.QueueLong.Node.SERIALIZER;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class LuceneCuvsBenchmarks {

  private static final Logger log = LoggerFactory.getLogger(LuceneCuvsBenchmarks.class.getName());
  private static final String PROACTIVE_SYNC_READY_FILE_PROPERTY =
      "cuvs.proactiveVectorSyncReadyFile";
  private static ExecutorService executorService = null;

  public enum Codex {
    LUCENE_HNSW,
    CAGRA_HNSW,
    CAGRA_SEARCH,
    CAGRA_HNSW_BINARY,
    CAGRA_HNSW_SCALAR
  }

  static KnnFloatVectorField createVectorField(BenchmarkConfiguration config, float[] vector) {
    return new KnnFloatVectorField(config.vectorColName, vector, config.vectorSimilarityFunction);
  }

  static Document createDocument(BenchmarkConfiguration config, int id, float[] vector) {
    Document document = new Document();
    if (!config.useLuceneDocIdAsVectorId) {
      document.add(new StringField("id", String.valueOf(id), Field.Store.YES));
    }
    document.add(createVectorField(config, vector));
    return document;
  }

  static int resolveVectorId(
      BenchmarkConfiguration config, int luceneDocId, String storedVectorId) {
    if (config.useLuceneDocIdAsVectorId) {
      return luceneDocId;
    }
    if (storedVectorId == null) {
      throw new IllegalStateException("Missing stored vector ID");
    }
    return Integer.parseInt(storedVectorId);
  }

  static boolean requiresGpuInitialization(BenchmarkConfiguration config) {
    if (config.isCagraSearch()) {
      return true;
    }
    return !config.skipIndexing
        && (config.isCagra() || config.isCagraHNSWBinary() || config.isCagraHNSWScalar());
  }

  static void validateMetricConfiguration(BenchmarkConfiguration config) {
    CuvsDistanceType expected = config.resolvedCuvsDistanceType();
    if (config.cagraIndexParamsStrategy == AcceleratedHNSWParams.Strategy.CUSTOM
        && config.cagraGraphBuildAlgo == CagraGraphBuildAlgo.IVF_PQ
        && config.cuVSIvfPqIndexParamsMetric != expected) {
      throw new IllegalArgumentException(
          "CUSTOM IVF-PQ metric must match vectorSimilarityFunction: expected "
              + expected
              + " but got "
              + config.cuVSIvfPqIndexParamsMetric);
    }
    if (config.cagraIndexParamsStrategy != AcceleratedHNSWParams.Strategy.CUSTOM) {
      config.cuVSIvfPqIndexParamsMetric = expected;
    }
    if (config.useLargeVectorWriteOutput && config.useCompoundFile) {
      throw new IllegalArgumentException(
          "useLargeVectorWriteOutput requires useCompoundFile=false");
    }
    if (config.largeVectorWriteBufferSizeKB < 1
        || config.largeVectorWriteBufferSizeKB > 64 * 1024) {
      throw new IllegalArgumentException(
          "largeVectorWriteBufferSizeKB must be between 1 and 65536");
    }
  }

  static Directory openIndexDirectory(BenchmarkConfiguration config, Path path) throws IOException {
    FSDirectory fsDirectory = FSDirectory.open(path);
    Directory output = fsDirectory;
    if (config.useLargeVectorWriteOutput) {
      int bufferBytes = Math.multiplyExact(config.largeVectorWriteBufferSizeKB, 1024);
      output = new LargeVectorWriteDirectory(fsDirectory, bufferBytes);
    }
    return new SyncTimingDirectory(output);
  }

  static void writeIngestCompleteMarker(Path marker, int indexedDocuments) throws IOException {
    Path parent = marker.toAbsolutePath().getParent();
    if (parent != null) {
      Files.createDirectories(parent);
    }
    String contents =
        "ingest_complete_epoch_ms\tindexed_documents\n"
            + System.currentTimeMillis()
            + "\t"
            + indexedDocuments
            + "\n";
    Files.writeString(marker, contents, StandardOpenOption.CREATE_NEW, StandardOpenOption.WRITE);
  }

  private static void signalIngestComplete(int indexedDocuments) throws IOException {
    String marker = System.getProperty(PROACTIVE_SYNC_READY_FILE_PROPERTY);
    if (marker == null || marker.isBlank()) {
      return;
    }
    writeIngestCompleteMarker(Path.of(marker), indexedDocuments);
  }

  /**
   * Uses reflection to bypass the 2048MB hard limit for per-thread RAM buffer.
   * This is a workaround to allow larger segments to be created before flushing.
   */
  private static void setPerThreadRAMLimit(IndexWriterConfig config, int limitMB) {
    try {
      // First, try to find the field in IndexWriterConfig
      java.lang.reflect.Field field = null;
      Class<?> clazz = config.getClass();

      // Try to find the field in the class hierarchy
      while (clazz != null && field == null) {
        try {
          field = clazz.getDeclaredField("perThreadHardLimitMB");
        } catch (NoSuchFieldException e) {
          // Try superclass
          clazz = clazz.getSuperclass();
        }
      }

      if (field == null) {
        // If not found in IndexWriterConfig, try LiveIndexWriterConfig
        clazz = config.getClass().getSuperclass();
        while (clazz != null && field == null) {
          try {
            field = clazz.getDeclaredField("perThreadHardLimitMB");
          } catch (NoSuchFieldException e) {
            clazz = clazz.getSuperclass();
          }
        }
      }

      if (field != null) {
        field.setAccessible(true);
        field.setInt(config, limitMB);
        log.info("Successfully set perThreadHardLimitMB to {} MB using reflection", limitMB);
      } else {
        log.error("Could not find perThreadHardLimitMB field using reflection");
      }
    } catch (Exception e) {
      log.error("Failed to set per-thread RAM limit using reflection", e);
    }
  }

  // ── Page cache management ─────────────────────────────────────────────────

  /**
   * Drops the OS page cache to ensure cold-cache measurements.
   * Requires root or passwordless sudo for tee.
   */
  private static void dropPageCache() {
    try {
      log.info("Dropping OS page cache...");
      ProcessBuilder pb =
          new ProcessBuilder("bash", "-c", "sync; echo 3 > /proc/sys/vm/drop_caches");
      pb.inheritIO();
      Process p = pb.start();
      int exit = p.waitFor();
      if (exit == 0) {
        log.info("Page cache dropped successfully");
      } else {
        log.warn("Failed to drop page cache (exit code {}). Are you running as root?", exit);
      }
    } catch (Exception e) {
      log.error("Failed to drop page cache", e);
    }
  }

  /**
   * Sequentially reads all files in the index directory to pull them into the
   * OS page cache. This ensures warm-cache measurements that are not dependent
   * on indexing-phase memory pressure.
   */
  private static void prewarmIndex(String indexDirPath) throws IOException {
    log.info("Pre-warming index files into page cache...");
    long start = System.currentTimeMillis();
    Path indexPath = Path.of(indexDirPath);
    byte[] buffer = new byte[1024 * 1024]; // 1MB read buffer
    try (var stream = Files.walk(indexPath)) {
      stream
          .filter(Files::isRegularFile)
          .forEach(
              file -> {
                try (InputStream fis = Files.newInputStream(file)) {
                  while (fis.read(buffer) != -1) {
                    // just reading — forces pages into OS cache
                  }
                } catch (IOException e) {
                  log.warn("Failed to prewarm file: {}", file, e);
                }
              });
    }
    log.info("Index pre-warm completed in {} ms", System.currentTimeMillis() - start);
  }

  public static void main(String[] args) throws Throwable {

    if (args.length < 1 || args.length > 3) {
      System.err.println("Usage: ./benchmarks.sh <jobs-file> [benchmarkID] [resultsDir]");
      return;
    }

    BenchmarkConfiguration config =
        Util.newObjectMapper().readValue(new File(args[0]), BenchmarkConfiguration.class);
    validateMetricConfiguration(config);

    // Override benchmarkID if provided as command line argument
    if (args.length >= 2) {
      config.benchmarkID = args[1];
    }

    // Override resultsDirectory if provided as command line argument
    if (args.length >= 3) {
      config.resultsDirectory = args[2];
    }
    Map<String, Object> metrics = new LinkedHashMap<String, Object>();
    List<QueryResult> queryResults = Collections.synchronizedList(new ArrayList<QueryResult>());
    config.debugPrintArguments();

    // [0] Pre-check
    Util.preCheck(config);

    String datasetMapdbFile = config.datasetFile + ".mapdb";

    // [1] Parse/load data set
    List<String> titles = new ArrayList<String>();
    VectorProvider vectorProvider;

    long parseStartTime = System.currentTimeMillis();

    // Check if dataset is .fvecs or .fbin format and handle it directly
    if (config.datasetFile.contains("fvecs") || config.datasetFile.contains("fbin")) {
      log.info("Detected .fvecs or .fbin file format. Loading directly without MapDB...");

      if (config.loadVectorsInMemory) {
        log.info("Loading all vectors in memory (loadVectorsInMemory is enabled)");
        long start = System.currentTimeMillis();
        List<float[]> loadedVectors = new ArrayList<float[]>();

        if (config.datasetFile.contains("fbin")) {
          FBIvecsReader.readFbin(config.datasetFile, config.numDocs, loadedVectors);
        } else {
          FBIvecsReader.readFvecs(config.datasetFile, config.numDocs, loadedVectors);
        }

        vectorProvider = new MemoryVectorProvider(loadedVectors);
        log.info(
            "Time taken to load {} vectors in-memory: {} ms",
            loadedVectors.size(),
            (System.currentTimeMillis() - start));
      } else if (ChunkedVectorProvider.supports(config.datasetFile)) {
        if (config.ingestPrefetch && config.numIndexThreads == 1) {
          log.info(
              "Creating prefetching chunked vector provider ({} MB chunks, double-buffered)",
              config.ingestChunkSizeMB);
          vectorProvider =
              new PrefetchingChunkedVectorProvider(
                  config.datasetFile, config.numDocs, config.ingestChunkSizeMB);
        } else {
          if (config.ingestPrefetch) {
            log.info(
                "Prefetch ingest requires numIndexThreads=1 (got {}); using sequential chunked"
                    + " provider",
                config.numIndexThreads);
          }
          log.info(
              "Creating chunked sequential vector provider ({} MB chunks)",
              config.ingestChunkSizeMB);
          vectorProvider =
              new ChunkedVectorProvider(
                  config.datasetFile, config.numDocs, config.ingestChunkSizeMB);
        }
      } else {
        log.info("Creating streaming vector provider (loadVectorsInMemory is disabled)");
        vectorProvider = new StreamingVectorProvider(config.datasetFile, config.numDocs);
      }

      titles.add(config.vectorColName);
    } else {
      // Use MapDB for non-.fvecs files (CSV, bvecs, etc.)
      DB db;
      IndexTreeList<float[]> vectors;

      if (new File(datasetMapdbFile).exists() == false) {
        log.info("Mapdb file not found for dataset. Preparing one ...");
        db = DBMaker.fileDB(datasetMapdbFile).make();
        vectors = db.indexTreeList("vectors", SERIALIZER.FLOAT_ARRAY).createOrOpen();
        if (config.datasetFile.endsWith(".csv") || config.datasetFile.endsWith(".csv.gz")) {
          Util.parseCSVFile(config, titles, vectors);
        } else if (config.datasetFile.contains("bvecs")) {
          Util.readBaseFile(config, titles, vectors);
        }
        log.info("Created a mapdb file with {} number of vectors.", vectors.size());
      } else {
        log.info("Mapdb file found for vectors. Loading ...");
        db = DBMaker.fileDB(datasetMapdbFile).make();
        vectors = db.indexTreeList("vectors", SERIALIZER.FLOAT_ARRAY).createOrOpen();
        log.info("{} vectors available from the mapdb file", vectors.size());
      }

      if (config.loadVectorsInMemory) {
        log.info(
            "Mapdb loaded. Now loading all vectors in memory (loadVectorsInMemory is enabled)");
        long start = System.currentTimeMillis();
        List<float[]> loadedVectors = new ArrayList<float[]>(vectors.size());
        for (int i = 0; i < vectors.size(); i++) {
          loadedVectors.add(vectors.get(i));
        }
        vectorProvider = new MemoryVectorProvider(loadedVectors);
        db.close();
        log.info(
            "Time taken to load the vectors in-memory is: {}",
            (System.currentTimeMillis() - start));
      } else {
        vectorProvider = new MapDBVectorProvider(vectors, db);
      }
    }
    long parseElapsedMs = System.currentTimeMillis() - parseStartTime;
    log.info("Time taken for parsing/loading dataset is {} ms", parseElapsedMs);
    StageTimers.record("dataset-load [DISK]", parseElapsedMs * 1_000_000L);

    try {
      // [2] Benchmarking setup
      // CAGRA_HNSW variants need the GPU while building, but their retained Lucene HNSW indexes
      // search on the CPU. CAGRA_SEARCH needs the GPU during search as well.
      if (requiresGpuInitialization(config)) {
        CuVSProvider.provider().enableRMMAsyncMemory();
        log.info("<<< cuVS provider and RMM async memory initialized >>>");
      }
      if (!config.skipIndexing) {
        int numVectorsToIndex = Math.min(config.numDocs, vectorProvider.size());

        // Partitioned K-segment build (K = numIndexThreads) when the whole dataset would otherwise
        // land in one segment (flushFreq >= numDocs) and no merge is requested. K sequential passes
        // each build one segment from a contiguous slice, so peak host memory stays at one slice's
        // native buffer; K=1 reduces to the single-segment path. This enables native flat buffering
        // and prefetch per segment, which the concurrent multi-thread path cannot (see
        // buildSegment).
        // Scoped to CAGRA_HNSW: sequential passes are free here because the GPU serializes builds
        // anyway. For a CPU build (LUCENE_HNSW) they would lose cross-segment parallelism.
        boolean partitioned =
            config.algoToRun.equals(Codex.CAGRA_HNSW)
                && !config.createIndexInMemory
                && config.forceMerge <= 0
                && !config.enableTieredMerge
                && config.flushFreq >= numVectorsToIndex;

        if (config.partitionedOverlap && !partitioned) {
          log.info(
              "partitionedOverlap ignored: applies only to the partitioned CAGRA_HNSW build"
                  + " (needs flushFreq>=numDocs, no merge, on-disk index)");
        }
        long indexStartTime = System.currentTimeMillis();
        if (partitioned) {
          if (config.partitionedOverlap
              && config.numIndexThreads > 1
              && PrefetchingChunkedVectorProvider.supports(config.datasetFile)) {
            buildPartitionedSegmentsOverlapped(config, numVectorsToIndex);
          } else {
            if (config.partitionedOverlap) {
              log.info(
                  "partitionedOverlap ignored: requires numIndexThreads>1 and a .fbin/.fvecs"
                      + " dataset (got numIndexThreads={})",
                  config.numIndexThreads);
            }
            buildPartitionedSegments(config, numVectorsToIndex, vectorProvider);
          }
        } else {
          IndexWriter writer;

          // HNSW Writer:
          IndexWriterConfig indexWriterConfig = new IndexWriterConfig(new StandardAnalyzer());
          indexWriterConfig.setCodec(getCodec(config, numVectorsToIndex));
          indexWriterConfig.setUseCompoundFile(config.useCompoundFile);
          indexWriterConfig.setMaxBufferedDocs(config.flushFreq);
          indexWriterConfig.setRAMBufferSizeMB(IndexWriterConfig.DISABLE_AUTO_FLUSH);

          if (config.forceMerge > 0 || config.enableTieredMerge) {
            TieredMergePolicy tmp = new TieredMergePolicy();
            if (config.forceMerge >= 1) {
              // With 10M x 1536-dim vectors the full index is ~60+ GiB.
              // Raise the ceiling above the full index size so the policy never
              // silently caps a forceMerge to the requested segment count.
              tmp.setMaxMergedSegmentMB(150 * 1024); // 150 GiB in MB
              tmp.setSegmentsPerTier(2);
              tmp.setMaxMergeAtOnce(500); // merge all segments in one round
            }
            indexWriterConfig.setMergePolicy(tmp);
          } else {
            indexWriterConfig.setMergePolicy(NoMergePolicy.INSTANCE);
          }

          // Use reflection to bypass the 2048MB per-thread limit and set it to 60GB
          setPerThreadRAMLimit(indexWriterConfig, 61440); // 60GB per thread
          log.info(
              "Configured HNSW writer - MaxBufferedDocs: {}, RAMBufferSizeMB: {},"
                  + " PerThreadRAMLimit: {} MB",
              config.flushFreq,
              indexWriterConfig.getRAMBufferSizeMB(),
              indexWriterConfig.getRAMPerThreadHardLimitMB());

          if (config.enableIndexWriterInfoStream) {
            indexWriterConfig.setInfoStream(new PrintStreamInfoStream(System.out));
          }

          if (!config.createIndexInMemory) {
            Path hnswIndex = Path.of(config.indexDirPath);
            writer = new IndexWriter(openIndexDirectory(config, hnswIndex), indexWriterConfig);
          } else {
            writer = new IndexWriter(new ByteBuffersDirectory(), indexWriterConfig);
          }

          log.info(
              "Indexing documents using {} ...",
              writer.getConfig().getCodec().knnVectorsFormat().getName());
          indexDocuments(writer, config, titles, vectorProvider);
        }
        long indexTimeTaken = System.currentTimeMillis() - indexStartTime;

        metrics.put(config.algoToRun + "-indexing-time", indexTimeTaken);
        StageTimers.appendMetrics(metrics, "benchmark-stage");
        log.info("Time taken for index building (end to end): {} ms", indexTimeTaken);

        try {
          if (!config.createIndexInMemory) {
            Path indexPath = Paths.get(config.indexDirPath);
            long directorySize;
            try (var stream = Files.walk(indexPath, FileVisitOption.FOLLOW_LINKS)) {
              directorySize =
                  stream.filter(p -> p.toFile().isFile()).mapToLong(p -> p.toFile().length()).sum();
            }

            double directorySizeGB = directorySize / 1_073_741_824.0;
            metrics.put(config.algoToRun + "-index-size", directorySizeGB);
            log.info("Size of {}: {} GB", indexPath.toString(), directorySizeGB);
          }
        } catch (IOException e) {
          log.error("Failed to calculate directory size for {}", config.indexDirPath, e);
        }
      }

      // ── Shrink JVM heap before search ──────────────────────────────────────
      // After indexing, the DWPT buffers are garbage. Trigger GC so the JVM
      // returns that physical RAM to the OS, making it available for the page
      // cache during search.
      log.info("Triggering GC to release heap before search...");
      System.gc();
      Thread.sleep(5000);
      log.info(
          "Heap after GC: committed={}MB, used={}MB",
          Runtime.getRuntime().totalMemory() / (1024 * 1024),
          (Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()) / (1024 * 1024));

      // Resolve the list of efSearch values to iterate over.
      // The index is built once above; we search it once per efSearch value.
      List<Integer> efSearchValues = config.getEfSearchValues();
      log.info(
          "Will run search with {} efSearch value(s): {}", efSearchValues.size(), efSearchValues);

      // Read ground truth once (shared across all efSearch runs)
      List<int[]> groundTruth = Util.readGroundTruthFile(config.groundTruthFile);

      Directory indexDir = MMapDirectory.open(Path.of(config.indexDirPath));
      log.info("Index directory is: {} (using memory-mapped files)", indexDir);

      // Snapshot indexing-only metrics before the loop so that each efSearch run
      // starts from the same base and doesn't inherit results from prior runs.
      Map<String, Object> indexingMetrics = new LinkedHashMap<>(metrics);

      for (int efSearch : efSearchValues) {
        log.info("--- Running search with efSearch={} ---", efSearch);

        // ── Deterministic cache state for each efSearch run ──────────────────
        if (config.dropPageCacheBeforeSearch) {
          dropPageCache();
        }
        if (config.prewarmIndexBeforeSearch) {
          prewarmIndex(config.indexDirPath);
        }

        // Fresh collections for this efSearch run
        List<QueryResult> efSearchQueryResults =
            Collections.synchronizedList(new ArrayList<QueryResult>());
        Map<String, Object> efSearchMetrics = new LinkedHashMap<String, Object>();

        // Copy over indexing metrics (only) so they appear in every result file
        efSearchMetrics.putAll(indexingMetrics);
        efSearchMetrics.put("efSearch", efSearch);

        log.info("Querying documents using {} with efSearch={} ...", config.algoToRun, efSearch);
        search(indexDir, config, efSearchMetrics, efSearchQueryResults, groundTruth, efSearch);

        Util.calculateRecallAccuracy(efSearchQueryResults, efSearchMetrics, config.algoToRun);

        String resultsJson =
            Util.newObjectMapper()
                .writerWithDefaultPrettyPrinter()
                .writeValueAsString(Map.of("configuration", config, "metrics", efSearchMetrics));

        if (config.saveResultsOnDisk) {
          // Use the resultsDirectory directly if provided
          String resultsDir = config.resultsDirectory != null ? config.resultsDirectory : "results";

          // When there are multiple efSearch values, create a subdirectory per value
          if (efSearchValues.size() > 1) {
            resultsDir = resultsDir + "/efSearch_" + efSearch;
          }

          File results = new File(resultsDir);
          if (!results.exists()) {
            results.mkdirs();
          }

          // Save results.json directly to the specified directory
          FileUtils.write(
              new File(results.toString() + "/results.json"),
              resultsJson,
              Charset.forName("UTF-8"));

          // Save CSV with neighbors data
          Util.writeCSV(efSearchQueryResults, results.toString() + "/neighbors.csv");

          log.info("Results for efSearch={} saved to directory: {}", efSearch, resultsDir);
        }

        log.info(
            "\n-----\nMetrics for efSearch={}: {}\n{}\n-----",
            efSearch,
            efSearchMetrics,
            resultsJson);

        // Accumulate per-efSearch metrics into the top-level metrics map
        for (Map.Entry<String, Object> entry : efSearchMetrics.entrySet()) {
          if (efSearchValues.size() > 1) {
            metrics.put("efSearch_" + efSearch + "/" + entry.getKey(), entry.getValue());
          } else {
            metrics.put(entry.getKey(), entry.getValue());
          }
        }
      }

      log.info("\n-----\nOverall metrics: {}\n-----", metrics);

      // Close the index directory before cleaning
      indexDir.close();

      // Clean index directory after benchmarks complete if requested
      if (config.cleanIndexDirectory && !config.createIndexInMemory) {
        Path indexPath = Path.of(config.indexDirPath);

        if (indexPath != null) {
          try {
            log.info("Cleaning index directory: {}", indexPath);
            FileUtils.deleteDirectory(indexPath.toFile());
            log.info("Successfully cleaned index directory: {}", indexPath);
          } catch (IOException e) {
            log.error("Failed to clean index directory: {}", indexPath, e);
          }
        }
      }
    } finally {
      if (vectorProvider != null) {
        vectorProvider.close();
      }
    }
    if (executorService != null && !executorService.isShutdown()) {
      executorService.shutdown();
      executorService.close();
      // Need the following as a temporary fix for now as the process gets stuck in case of
      // LUCENE_HNSW when using multiple merge threads
      System.exit(0);
    }
  }

  static void indexDocuments(
      IndexWriter writer,
      BenchmarkConfiguration config,
      List<String> titles,
      VectorProvider vectorProvider)
      throws IOException, InterruptedException {

    int threads = config.numIndexThreads;
    ExecutorService pool = Executors.newFixedThreadPool(threads);
    AtomicInteger numDocsIndexed = new AtomicInteger(0);
    log.info("Starting indexing with {} threads.", threads);
    log.info(
        "IndexWriter config - MaxBufferedDocs: {}, RAMBufferSizeMB: {}",
        writer.getConfig().getMaxBufferedDocs(),
        writer.getConfig().getRAMBufferSizeMB());
    final int numDocsToIndex = Math.min(config.numDocs, vectorProvider.size());

    long ingestStart = StageTimers.start();
    // The prefetch provider unpacks directly into a caller-reused array (safe: the writer copies
    // each vector eagerly at addValue). Other providers keep returning fresh arrays.
    final boolean reuseScratch = vectorProvider instanceof PrefetchingChunkedVectorProvider;
    List<Future<?>> ingestTasks = new ArrayList<>(threads);
    for (int i = 0; i < threads; i++) {
      ingestTasks.add(
          pool.submit(
              () -> {
                float[] scratch = reuseScratch ? new float[config.vectorDimension] : null;
                while (true) {
                  int id = numDocsIndexed.getAndIncrement();
                  if (id >= numDocsToIndex) {
                    break; // done
                  }
                  float[] vector;
                  try {
                    if (reuseScratch) {
                      vectorProvider.get(id, scratch);
                      vector = scratch;
                    } else {
                      vector = Objects.requireNonNull(vectorProvider.get(id));
                    }
                  } catch (IOException e) {
                    throw new UncheckedIOException("Failed to read vector at index " + id, e);
                  }
                  Document doc = createDocument(config, id, vector);
                  try {
                    writer.addDocument(doc);
                    if ((id + 1) % 25000 == 0) {
                      log.info(
                          "Done indexing {} documents. Pending docs: {}",
                          (id + 1),
                          writer.getPendingNumDocs());
                    }
                    // Log when we expect a flush
                    if ((id + 1) == config.flushFreq || (id + 1) == 2 * config.flushFreq) {
                      log.info("Expected flush point reached at {} documents", (id + 1));
                    }
                  } catch (IOException ex) {
                    throw new UncheckedIOException(ex);
                  }
                }
              }));
    }
    awaitIngestTasks(pool, ingestTasks);
    long ingestBytes = (long) numDocsToIndex * config.vectorDimension * Float.BYTES;
    StageTimers.stop("ingest [DISK+CPU]", ingestStart, ingestBytes);
    signalIngestComplete(numDocsToIndex);

    if (config.forceMerge > 0) {
      log.info("Force merge is enabled, force merging into " + config.forceMerge + " segments");
      long forceMergeStart = StageTimers.start();
      writer.forceMerge(config.forceMerge);
      StageTimers.stop("force-merge [GPU+CPU+DISK]", forceMergeStart);
    }

    log.info("Calling commit.");
    long commitStart = StageTimers.start();
    writer.commit();
    StageTimers.stop("commit [DISK; flush + GPU build + fsync]", commitStart);
    long closeStart = StageTimers.start();
    writer.close();
    StageTimers.stop("close [DISK; finalize]", closeStart);
  }

  static void awaitIngestTasks(ExecutorService pool, List<? extends Future<?>> tasks)
      throws IOException, InterruptedException {
    pool.shutdown();
    Throwable firstFailure = null;
    try {
      for (Future<?> task : tasks) {
        try {
          task.get();
        } catch (ExecutionException e) {
          if (firstFailure == null) {
            firstFailure = e.getCause();
          }
        }
      }
    } catch (InterruptedException e) {
      for (Future<?> task : tasks) {
        task.cancel(true);
      }
      pool.shutdownNow();
      Thread.currentThread().interrupt();
      throw e;
    }

    if (firstFailure instanceof UncheckedIOException uncheckedIOException) {
      throw uncheckedIOException.getCause();
    }
    if (firstFailure instanceof IOException ioException) {
      throw ioException;
    }
    if (firstFailure instanceof RuntimeException runtimeException) {
      throw runtimeException;
    }
    if (firstFailure instanceof Error error) {
      throw error;
    }
    if (firstFailure != null) {
      throw new RuntimeException("Ingest worker failed", firstFailure);
    }
  }

  /**
   * Partitioned K-segment build: {@code K = numIndexThreads} contiguous slices, each built
   * sequentially as one segment appended to the same directory. Sequential (not concurrent) so peak
   * host memory is a single slice's native buffer, not K of them. The vector provider is streamed
   * front-to-back across the passes, so a single forward-only prefetch reader spans all K.
   */
  private static void buildPartitionedSegments(
      BenchmarkConfiguration config, int numVectorsToIndex, VectorProvider vectorProvider)
      throws Exception {
    int k = Math.max(1, config.numIndexThreads);
    log.info(
        "Partitioned build: {} segment(s) from {} vectors (K = numIndexThreads), sequential passes",
        k,
        numVectorsToIndex);
    // One directory shared across all K passes: its sync-once tracking skips re-fsyncing prior
    // segments that each fresh per-pass IndexWriter would otherwise re-sync (O(K^2) -> O(K)).
    try (Directory dir = openIndexDirectory(config, Path.of(config.indexDirPath))) {
      int base = numVectorsToIndex / k;
      int rem = numVectorsToIndex % k;
      int start = 0;
      for (int p = 0; p < k; p++) {
        int size = base + (p < rem ? 1 : 0); // spread the remainder over the first slices
        if (size <= 0) {
          continue;
        }
        log.info("Building segment {}/{}: docs [{}, {})", p + 1, k, start, start + size);
        buildSegment(config, dir, start, size, vectorProvider, p == 0);
        start += size;
      }
    }
  }

  /**
   * Overlapped variant of the partitioned build (prototype). Each of the K = numIndexThreads
   * segments is built into its own directory by a bounded pool ({@code partitionedPipelineDepth}
   * concurrent tasks), so a segment's ingest overlaps a prior segment's GPU commit; the GPU build
   * is serialized on a single permit. Peak host memory is {@code depth * (N/K)} native buffers.
   * The K per-segment indexes are then assembled into the final directory by hardlinking their
   * files (same filesystem) via addIndexes — no bulk copy of the vector data.
   */
  private static void buildPartitionedSegmentsOverlapped(
      BenchmarkConfiguration config, int numVectorsToIndex) throws Exception {
    int k = Math.max(1, config.numIndexThreads);
    int depth = Math.max(1, Math.min(k, config.partitionedPipelineDepth));

    int base = numVectorsToIndex / k;
    int rem = numVectorsToIndex % k;
    List<int[]> slices = new ArrayList<>(); // [start, size] per segment
    List<Path> segDirs = new ArrayList<>();
    int start = 0;
    int maxSlice = 0;
    for (int p = 0; p < k; p++) {
      int size = base + (p < rem ? 1 : 0);
      if (size <= 0) {
        continue;
      }
      slices.add(new int[] {start, size});
      segDirs.add(Path.of(config.indexDirPath + "_p" + p));
      start += size;
      maxSlice = Math.max(maxSlice, size);
    }

    // Peak HOST memory is up to `depth` co-resident native buffers (each ~ one slice's vectors);
    // DEVICE memory stays at a single build's footprint because the GPU commit is serialized on
    // gpuPermit (host-side ingest never touches the device).
    double peakHostGb = (double) depth * maxSlice * config.vectorDimension * Float.BYTES / 1e9;
    log.info(
        "Partitioned build (overlapped): {} segment(s) from {} vectors, pipeline depth {}"
            + " (up to {} co-resident native host buffers, ~{} GB peak host)",
        slices.size(),
        numVectorsToIndex,
        depth,
        depth,
        String.format("%.1f", peakHostGb));

    // Start from fresh per-segment temp dirs, and always remove them afterwards (even on failure)
    // so a crashed build does not leave ~N GB of orphaned indexes behind.
    for (Path segDir : segDirs) {
      FileUtils.deleteQuietly(segDir.toFile());
    }
    try {
      Semaphore gpuPermit = new Semaphore(1);
      ExecutorService pool = Executors.newFixedThreadPool(depth);
      List<Future<?>> futures = new ArrayList<>();
      for (int i = 0; i < slices.size(); i++) {
        final int[] slice = slices.get(i);
        final Path segDir = segDirs.get(i);
        futures.add(
            pool.submit(
                () -> {
                  Directory d = new SyncTimingDirectory(FSDirectory.open(segDir));
                  VectorProvider prov =
                      new PrefetchingChunkedVectorProvider(
                          config.datasetFile, slice[0], slice[1], config.ingestChunkSizeMB);
                  try {
                    buildSegment(config, d, slice[0], slice[1], prov, true, gpuPermit);
                  } finally {
                    prov.close();
                    d.close();
                  }
                  return null;
                }));
      }
      pool.shutdown();
      try {
        for (Future<?> f : futures) {
          f.get(); // propagate any build failure
        }
      } finally {
        pool.shutdownNow();
      }

      // Assemble the per-segment indexes into the final directory. Hardlink their files (same FS)
      // instead of copying the ~N GB of vector data; HardlinkCopyDirectoryWrapper falls back to a
      // byte copy if the segment dirs and the final dir are on different filesystems.
      long combineStart = StageTimers.start();
      Directory[] sources = new Directory[segDirs.size()];
      try {
        for (int i = 0; i < segDirs.size(); i++) {
          sources[i] = FSDirectory.open(segDirs.get(i));
        }
        IndexWriterConfig iwc =
            new IndexWriterConfig(new StandardAnalyzer()).setMergePolicy(NoMergePolicy.INSTANCE);
        try (Directory target =
                new HardlinkCopyDirectoryWrapper(FSDirectory.open(Path.of(config.indexDirPath)));
            IndexWriter combiner = new IndexWriter(target, iwc)) {
          combiner.addIndexes(sources);
        }
      } finally {
        for (Directory s : sources) {
          if (s != null) {
            s.close();
          }
        }
      }
      StageTimers.stop("combine [addIndexes+hardlink]", combineStart);
    } finally {
      for (Path segDir : segDirs) {
        FileUtils.deleteQuietly(segDir.toFile());
      }
    }
  }

  /**
   * Builds one segment from the contiguous slice {@code [sliceStart, sliceStart + sliceSize)}: a
   * fresh single-threaded {@link IndexWriter} whose codec's {@code numInputVectors} is sized for the
   * slice (so native flat buffering applies per segment). The first pass opens with {@code CREATE}
   * (fresh index); later passes {@code APPEND} their segment. {@link NoMergePolicy} keeps the K
   * segments separate. The provider is consumed forward, matching the prefetch reader's contract.
   */
  private static void buildSegment(
      BenchmarkConfiguration config,
      Directory directory,
      int sliceStart,
      int sliceSize,
      VectorProvider vectorProvider,
      boolean createNew)
      throws Exception {
    buildSegment(config, directory, sliceStart, sliceSize, vectorProvider, createNew, null);
  }

  /**
   * As {@link #buildSegment(BenchmarkConfiguration, Directory, int, int, VectorProvider, boolean)},
   * but when {@code gpuCommitPermit} is non-null the commit (which runs the GPU CAGRA build) is
   * serialized on it while other segments' ingest may proceed — the partitioned-overlap path.
   */
  private static void buildSegment(
      BenchmarkConfiguration config,
      Directory directory,
      int sliceStart,
      int sliceSize,
      VectorProvider vectorProvider,
      boolean createNew,
      Semaphore gpuCommitPermit)
      throws Exception {
    IndexWriterConfig iwc = new IndexWriterConfig(new StandardAnalyzer());
    iwc.setCodec(getCodec(config, sliceSize)); // numInputVectors = sliceSize for this segment
    iwc.setUseCompoundFile(config.useCompoundFile);
    iwc.setMaxBufferedDocs(config.flushFreq);
    iwc.setRAMBufferSizeMB(IndexWriterConfig.DISABLE_AUTO_FLUSH);
    iwc.setMergePolicy(NoMergePolicy.INSTANCE);
    iwc.setOpenMode(
        createNew ? IndexWriterConfig.OpenMode.CREATE : IndexWriterConfig.OpenMode.APPEND);
    setPerThreadRAMLimit(iwc, 61440); // 60GB per thread
    if (config.enableIndexWriterInfoStream) {
      iwc.setInfoStream(new PrintStreamInfoStream(System.out));
    }

    IndexWriter writer = new IndexWriter(directory, iwc);
    boolean reuseScratch = vectorProvider instanceof PrefetchingChunkedVectorProvider;
    float[] scratch = reuseScratch ? new float[config.vectorDimension] : null;
    try {
      long ingestStart = StageTimers.start();
      for (int i = 0; i < sliceSize; i++) {
        int id = sliceStart + i;
        float[] vector;
        try {
          if (reuseScratch) {
            vectorProvider.get(id, scratch);
            vector = scratch;
          } else {
            vector = Objects.requireNonNull(vectorProvider.get(id));
          }
        } catch (IOException e) {
          throw new UncheckedIOException("Failed to read vector at index " + id, e);
        }
        Document doc = createDocument(config, id, vector);
        writer.addDocument(doc);
        if ((id + 1) % 25000 == 0) {
          log.info(
              "Done indexing {} documents. Pending docs: {}", id + 1, writer.getPendingNumDocs());
        }
      }
      StageTimers.stop(
          "ingest [DISK+CPU]",
          ingestStart,
          (long) sliceSize * config.vectorDimension * Float.BYTES);
      signalIngestComplete(sliceSize);
      if (gpuCommitPermit != null) {
        gpuCommitPermit.acquire(); // serialize the GPU build across concurrent segment pipelines
      }
      try {
        long commitStart = StageTimers.start();
        writer.commit();
        StageTimers.stop("commit [DISK; flush + GPU build + fsync]", commitStart);
      } finally {
        if (gpuCommitPermit != null) {
          gpuCommitPermit.release();
        }
      }
    } finally {
      long closeStart = StageTimers.start();
      writer.close();
      StageTimers.stop("close [DISK; finalize]", closeStart);
    }
  }

  /**
   * Runs search queries against the given index directory using the specified efSearch value.
   *
   * @param directory    the Lucene index directory to search
   * @param config       benchmark configuration
   * @param metrics      map to populate with search performance metrics
   * @param queryResults list to populate with per-query results
   * @param groundTruth  ground truth neighbor lists for recall calculation
   * @param efSearch     the efSearch (number of candidates) to use for this search run
   */
  private static void search(
      Directory directory,
      BenchmarkConfiguration config,
      Map<String, Object> metrics,
      List<QueryResult> queryResults,
      List<int[]> groundTruth,
      int efSearch) {

    DB db = null;
    try (IndexReader indexReader = DirectoryReader.open(directory)) {
      IndexSearcher indexSearcher = new IndexSearcher(indexReader);

      List<float[]> queries;
      String queryMapdbFile = config.queryFile + ".mapdb";

      if (isExplicitBinaryQueryFile(config.queryFile)) {
        queries = readExplicitBinaryQueries(config.queryFile, config.numQueriesToRun);
        log.info("Read {} queries directly from {}", queries.size(), config.queryFile);
      } else if (!isLegacyCachedQueryFile(config.queryFile)) {
        throw new IllegalArgumentException("Unsupported query file: " + config.queryFile);
      } else if (new File(queryMapdbFile).exists() == false) {
        log.info("No mapdb file found for queries. Reading source files to build one ...");
        db = DBMaker.fileDB(queryMapdbFile).make();
        IndexTreeList<float[]> cachedQueries =
            db.indexTreeList("vectors", SERIALIZER.FLOAT_ARRAY).createOrOpen();
        queries = cachedQueries;

        if (config.queryFile.endsWith(".csv")) {
          for (String line :
              FileUtils.readFileToString(new File(config.queryFile), "UTF-8").split("\n")) {
            queries.add(Util.parseFloatArrayFromStringArray(line));
          }
        } else if (config.queryFile.endsWith(".bvecs")
            || config.queryFile.endsWith(".bvecs.gz")) {
          FBIvecsReader.readBvecs(config.queryFile, -1, queries);
        }
        log.info("Mapdb file created with {} number of queries", queries.size());
      } else {
        log.info("Mapdb file found for queries. Loading ...");
        db = DBMaker.fileDB(queryMapdbFile).make();
        queries = db.indexTreeList("vectors", SERIALIZER.FLOAT_ARRAY).createOrOpen();
        log.info("{} queries available from the mapdb file", queries.size());
      }
      validateQueries(queries, config.numQueriesToRun, config.vectorDimension, config.queryFile);

      ExecutorService pool = Executors.newFixedThreadPool(config.queryThreads);
      AtomicInteger queriesFinished = new AtomicInteger(0);
      ConcurrentHashMap<Integer, Double> queryLatencies = new ConcurrentHashMap<Integer, Double>();
      ConcurrentHashMap<Integer, Double> retrievalLatencies =
          new ConcurrentHashMap<Integer, Double>();

      long startTime = System.currentTimeMillis();
      AtomicInteger queryId = new AtomicInteger(0);

      for (int t = 0; t < config.queryThreads; t++) {
        pool.submit(
            () -> {
              while (true) {
                int currentQueryId = queryId.getAndIncrement();
                if (currentQueryId >= config.numQueriesToRun) {
                  break;
                }
                try {
                  KnnFloatVectorQuery query;

                  if (config.algoToRun.equals(Codex.CAGRA_SEARCH)) {
                    query =
                        new GPUKnnFloatVectorQuery(
                            config.vectorColName,
                            queries.get(currentQueryId),
                            efSearch,
                            null,
                            config.cagraITopK,
                            config.cagraSearchWidth);
                  } else {
                    query =
                        new KnnFloatVectorQuery(
                            config.vectorColName, queries.get(currentQueryId), efSearch);
                  }

                  TopDocs topDocs;
                  long searchStartTime = System.nanoTime();
                  try {
                    TopScoreDocCollectorManager collectorManager =
                        new TopScoreDocCollectorManager(efSearch, null, Integer.MAX_VALUE, true);
                    topDocs = indexSearcher.search(query, collectorManager);
                  } catch (IOException e) {
                    throw new RuntimeException("Problem during executing a query: ", e);
                  }
                  double searchTimeTakenMs = (System.nanoTime() - searchStartTime) / 1_000_000.0;
                  if (currentQueryId >= config.numWarmUpQueries) {
                    queryLatencies.put(currentQueryId, searchTimeTakenMs);
                  }
                  int finishedCount = queriesFinished.incrementAndGet();

                  // Log progress every 1000 queries
                  if (finishedCount % 1000 == 0 || finishedCount == config.numQueriesToRun) {
                    log.info(
                        "Done querying "
                            + finishedCount
                            + " out of "
                            + config.numQueriesToRun
                            + " queries.");
                  }

                  ScoreDoc[] hits = topDocs.scoreDocs;
                  List<Integer> neighbors = new ArrayList<>();
                  List<Float> scores = new ArrayList<>();

                  // Debug: Log search results for first query
                  if (queryId.get() == 0) {
                    log.info(
                        "Debug: First query returned "
                            + hits.length
                            + " hits (ef-search candidates)");
                    log.info(
                        "Debug: Will select top "
                            + config.topK
                            + " from "
                            + hits.length
                            + " candidates");
                  }
                  int numResultsToTake = Math.min(config.topK, hits.length);
                  long retrievalStartTime = System.nanoTime();
                  for (int i = 0; i < numResultsToTake; i++) {
                    ScoreDoc hit = hits[i];
                    try {
                      String storedVectorId = null;
                      if (!config.useLuceneDocIdAsVectorId) {
                        Document d = indexReader.storedFields().document(hit.doc);
                        storedVectorId = d.get("id");
                      }
                      neighbors.add(resolveVectorId(config, hit.doc, storedVectorId));
                    } catch (IOException e) {
                      e.printStackTrace();
                    }
                    scores.add(hit.score);
                  }
                  double retrievalTimeTakenMs =
                      (System.nanoTime() - retrievalStartTime) / 1_000_000.0;
                  if (currentQueryId >= config.numWarmUpQueries) {
                    retrievalLatencies.put(currentQueryId, retrievalTimeTakenMs);
                  }

                  // Debug: Log results for all queries
                  log.debug(
                      "Query "
                          + currentQueryId
                          + " - First 5 neighbors: "
                          + neighbors.subList(0, Math.min(5, neighbors.size())));
                  log.debug(
                      "Query "
                          + currentQueryId
                          + " - First 5 distances: "
                          + scores.subList(0, Math.min(5, scores.size())));
                  int[] expectedNeighbors = groundTruth.get(currentQueryId);
                  log.debug(
                      "Query "
                          + currentQueryId
                          + " - Expected neighbors: "
                          + java.util.Arrays.toString(
                              java.util.Arrays.copyOf(
                                  expectedNeighbors, Math.min(5, expectedNeighbors.length))));

                  if (currentQueryId >= config.numWarmUpQueries) {
                    QueryResult result =
                        new QueryResult(
                            config.algoToRun.toString(),
                            currentQueryId,
                            neighbors,
                            groundTruth.get(currentQueryId),
                            scores,
                            searchTimeTakenMs);
                    queryResults.add(result);
                  } else {
                    log.info("Skipping warmup query: {}", currentQueryId);
                  }
                } catch (Exception e) {
                  log.error(
                      "Exception during query {}: {} - {}",
                      currentQueryId,
                      e.getClass().getSimpleName(),
                      e.getMessage(),
                      e);
                }
              }
            });
      }

      pool.shutdown();
      pool.awaitTermination(Long.MAX_VALUE, TimeUnit.SECONDS);

      long endTime = System.currentTimeMillis();

      metrics.put(config.algoToRun + "-query-time", (endTime - startTime));
      metrics.put(
          config.algoToRun + "-query-throughput",
          (endTime - startTime) > 0
              ? (config.numQueriesToRun / ((endTime - startTime) / 1000.0))
              : 0.0);
      double avgLatency =
          queryLatencies.isEmpty()
              ? 0.0
              : new ArrayList<>(queryLatencies.values()).stream().reduce(0.0, Double::sum)
                  / queryLatencies.size();
      double avgRetLatency =
          retrievalLatencies.isEmpty()
              ? 0.0
              : new ArrayList<>(retrievalLatencies.values()).stream().reduce(0.0, Double::sum)
                  / retrievalLatencies.size();

      metrics.put(config.algoToRun + "-mean-latency", avgLatency);
      metrics.put(config.algoToRun + "-mean-retrieval-latency", avgRetLatency);

      // Log warning if no queries completed successfully
      if (queryLatencies.isEmpty()) {
        log.error(
            "WARNING: Zero queries completed successfully! "
                + "Check the query thread exception logs above for the root cause. "
                + "queriesFinished={}, queryResults.size={}",
            queriesFinished.get(),
            queryResults.size());
      }

      int segmentCount = indexReader.leaves().size();
      metrics.put(config.algoToRun + "-segment-count", segmentCount);

    } catch (Exception e) {
      e.printStackTrace();
      log.error("Exception during querying", e);
    } finally {
      if (db != null) {
        db.close();
      }
    }
  }

  static boolean isExplicitBinaryQueryFile(String queryFile) {
    return queryFile.endsWith(".fbin")
        || queryFile.endsWith(".fvecs")
        || queryFile.endsWith(".fvecs.gz");
  }

  private static boolean isLegacyCachedQueryFile(String queryFile) {
    return queryFile.endsWith(".csv")
        || queryFile.endsWith(".bvecs")
        || queryFile.endsWith(".bvecs.gz");
  }

  static List<float[]> readExplicitBinaryQueries(String queryFile, int requiredCount) {
    if (!isExplicitBinaryQueryFile(queryFile)) {
      throw new IllegalArgumentException("Not an explicit binary query file: " + queryFile);
    }
    List<float[]> queries = new ArrayList<>();
    if (queryFile.endsWith(".fbin")) {
      FBIvecsReader.readFbin(queryFile, requiredCount, queries);
    } else {
      FBIvecsReader.readFvecs(queryFile, requiredCount, queries);
    }
    return queries;
  }

  static void validateQueries(
      List<float[]> queries, int requiredCount, int expectedDimension, String queryFile) {
    if (queries.size() < requiredCount) {
      throw new IllegalArgumentException(
          "Query file "
              + queryFile
              + " contains "
              + queries.size()
              + " vectors; "
              + requiredCount
              + " required");
    }
    for (int i = 0; i < queries.size(); i++) {
      if (queries.get(i).length != expectedDimension) {
        throw new IllegalArgumentException(
            "Query "
                + i
                + " in "
                + queryFile
                + " has dimension "
                + queries.get(i).length
                + "; expected "
                + expectedDimension);
      }
    }
  }

  static CuVSIvfPqParams buildCustomIvfPqParams(
      BenchmarkConfiguration config, int numVectorsToIndex) {
    validateCustomIvfPqParams(config, numVectorsToIndex);
    CuVSIvfPqIndexParams indexParams =
        new CuVSIvfPqIndexParams.Builder()
            .withMetric(config.cuVSIvfPqIndexParamsMetric)
            .withMetricArg(config.cuVSIvfPqIndexParamsMetricArg)
            .withAddDataOnBuild(config.cuVSIvfPqIndexParamsAddDataOnBuild)
            .withNLists(config.cuVSIvfPqIndexParamsNLists)
            .withKmeansNIters(config.cuVSIvfPqIndexParamsKmeansNIters)
            .withKmeansTrainsetFraction(config.cuVSIvfPqIndexParamsKmeansTrainsetFraction)
            .withPqBits(config.cuVSIvfPqIndexParamsPqBits)
            .withPqDim(config.cuVSIvfPqIndexParamsPqDim)
            .withCodebookKind(config.cuVSIvfPqIndexParamsCodebookKind)
            .withForceRandomRotation(config.cuVSIvfPqIndexParamsForceRandomRotation)
            .withConservativeMemoryAllocation(
                config.cuVSIvfPqIndexParamsConservativeMemoryAllocation)
            .withMaxTrainPointsPerPqCode(config.cuVSIvfPqIndexParamsMaxTrainPointsPerPqCode)
            .build();
    CuVSIvfPqSearchParams searchParams =
        new CuVSIvfPqSearchParams.Builder()
            .withNProbes(config.cuVSIvfPqSearchParamsNProbes)
            .withLutDtype(config.cuVSIvfPqSearchParamsLutDtype)
            .withInternalDistanceDtype(config.cuVSIvfPqSearchParamsInternalDistanceDtype)
            .withPreferredShmemCarveout(config.cuVSIvfPqSearchParamsPreferredShmemCarveout)
            .build();
    return new CuVSIvfPqParams.Builder()
        .withCuVSIvfPqIndexParams(indexParams)
        .withCuVSIvfPqSearchParams(searchParams)
        .withRefinementRate(config.cuVSIvfPqParamsRefinementRate)
        .build();
  }

  private static void validateCustomIvfPqParams(
      BenchmarkConfiguration config, int numVectorsToIndex) {
    if (config.cuVSIvfPqIndexParamsMetric != config.resolvedCuvsDistanceType()) {
      throw new IllegalArgumentException(
          "CUSTOM IVF-PQ metric must match vectorSimilarityFunction");
    }
    if (config.cuVSIvfPqIndexParamsNLists < 1
        || config.cuVSIvfPqIndexParamsNLists > numVectorsToIndex) {
      throw new IllegalArgumentException("IVF-PQ nLists must be within the indexed vector count");
    }
    if (config.cuVSIvfPqSearchParamsNProbes < 1
        || config.cuVSIvfPqSearchParamsNProbes > config.cuVSIvfPqIndexParamsNLists) {
      throw new IllegalArgumentException("IVF-PQ nProbes must be between 1 and nLists");
    }
    if (config.cuVSIvfPqIndexParamsPqBits < 4
        || config.cuVSIvfPqIndexParamsPqBits > 8
        || config.cuVSIvfPqIndexParamsPqDim < 0
        || ((long) config.cuVSIvfPqIndexParamsPqDim * config.cuVSIvfPqIndexParamsPqBits) % 8 != 0) {
      throw new IllegalArgumentException("Invalid IVF-PQ pqBits/pqDim combination");
    }
    if (!Double.isFinite(config.cuVSIvfPqIndexParamsKmeansTrainsetFraction)
        || config.cuVSIvfPqIndexParamsKmeansTrainsetFraction <= 0.0
        || config.cuVSIvfPqIndexParamsKmeansTrainsetFraction > 1.0) {
      throw new IllegalArgumentException("IVF-PQ training fraction must be in (0, 1]");
    }
    if (config.cuVSIvfPqIndexParamsKmeansNIters < 1
        || config.cuVSIvfPqIndexParamsMaxTrainPointsPerPqCode < 1
        || config.cuVSIvfPqParamsRefinementRate < 1) {
      throw new IllegalArgumentException(
          "IVF-PQ iteration, training, and refinement values must be positive");
    }
    if (!config.cuVSIvfPqIndexParamsAddDataOnBuild) {
      throw new IllegalArgumentException("CAGRA graph construction requires addDataOnBuild=true");
    }
    if (config.cuVSIvfPqSearchParamsLutDtype == CudaDataType.CUDA_R_32F
        && config.cuVSIvfPqSearchParamsInternalDistanceDtype == CudaDataType.CUDA_R_16F) {
      throw new IllegalArgumentException(
          "Internal distance dtype cannot be narrower than LUT dtype");
    }
  }

  private static Codec getCodec(BenchmarkConfiguration config, int numVectorsToIndex)
      throws Exception {
    if (config.algoToRun.equals(Codex.LUCENE_HNSW)) {
      log.info("<<< Using Lucene104Codec >>>");
      return new Lucene104Codec(Mode.BEST_SPEED) {
        @Override
        public KnnVectorsFormat getKnnVectorsFormatForField(String field) {
          KnnVectorsFormat knnFormat;
          if (config.hnswMergeThreads > 1) {
            executorService = Executors.newFixedThreadPool(config.hnswMergeThreads);
            knnFormat =
                new Lucene99HnswVectorsFormat(
                    config.hnswMaxConn,
                    config.hnswBeamWidth,
                    config.hnswMergeThreads,
                    executorService);
          } else {
            knnFormat = new Lucene99HnswVectorsFormat(config.hnswMaxConn, config.hnswBeamWidth);
          }
          return new HighDimensionKnnVectorsFormat(knnFormat, config.vectorDimension);
        }
      };
    } else {
      AcceleratedHNSWParams.Strategy strategy =
          config.cagraIndexParamsStrategy != null
              ? config.cagraIndexParamsStrategy
              : AcceleratedHNSWParams.Strategy.HEURISTIC;
      AcceleratedHNSWParams.Builder paramsBuilder =
          new AcceleratedHNSWParams.Builder()
              .withStrategy(strategy)
              .withWriterThreads(config.cuvsWriterThreads)
              .withIntermediateGraphDegree(config.cagraIntermediateGraphDegree)
              .withGraphDegree(config.cagraGraphDegree)
              .withHNSWLayer(config.cagraHnswLayers)
              .withMaxConn(config.hnswMaxConn)
              .withBeamWidth(config.hnswBeamWidth)
              .withCagraGraphBuildAlgo(
                  config.cagraGraphBuildAlgo != null
                      ? config.cagraGraphBuildAlgo
                      : CagraGraphBuildAlgo.AUTO_SELECT)
              .withCuvsDistanceType(config.resolvedCuvsDistanceType());

      if (strategy == AcceleratedHNSWParams.Strategy.CUSTOM
          && config.cagraGraphBuildAlgo == CagraGraphBuildAlgo.IVF_PQ) {
        paramsBuilder.withCuVSIvfPqParams(buildCustomIvfPqParams(config, numVectorsToIndex));
      }

      AcceleratedHNSWParams params = paramsBuilder.build();
      log.info("<<< Resolved AcceleratedHNSWParams: {} >>>", params);

      if (config.algoToRun.equals(Codex.CAGRA_HNSW)) {
        log.info("<<< Using Lucene101AcceleratedHNSWCodec ({} strategy) >>>", strategy);
        return new Lucene101AcceleratedHNSWCodec(params);
      } else if (config.algoToRun.equals(Codex.CAGRA_SEARCH)) {
        log.info("<<< Using CuVS2510GPUSearchCodec (HEURISTIC strategy) >>>");
        GPUSearchParams gpuParams =
            new GPUSearchParams.Builder()
                .withStrategy(GPUSearchParams.Strategy.HEURISTIC)
                .withWriterThreads(config.cuvsWriterThreads)
                .withIntermediateGraphDegree(config.cagraIntermediateGraphDegree)
                .withGraphDegree(config.cagraGraphDegree)
                .build();
        return new CuVS2510GPUSearchCodec(gpuParams);
      } else if (config.algoToRun.equals(Codex.CAGRA_HNSW_BINARY)) {
        log.info("<<< Using LuceneAcceleratedHNSWBinaryQuantizedCodec (HEURISTIC strategy) >>>");
        return new LuceneAcceleratedHNSWBinaryQuantizedCodec(params);
      } else if (config.algoToRun.equals(Codex.CAGRA_HNSW_SCALAR)) {
        log.info("<<< Using LuceneAcceleratedHNSWScalarQuantizedCodec (HEURISTIC strategy) >>>");
        return new LuceneAcceleratedHNSWScalarQuantizedCodec(params);
      }
    }
    return null;
  }

  private static class HighDimensionKnnVectorsFormat extends KnnVectorsFormat {
    private final KnnVectorsFormat knnFormat;
    private final int maxDimensions;

    public HighDimensionKnnVectorsFormat(KnnVectorsFormat knnFormat, int maxDimensions) {
      super(knnFormat.getName());
      this.knnFormat = knnFormat;
      this.maxDimensions = maxDimensions;
    }

    @Override
    public KnnVectorsWriter fieldsWriter(SegmentWriteState state) throws IOException {
      return knnFormat.fieldsWriter(state);
    }

    @Override
    public KnnVectorsReader fieldsReader(SegmentReadState state) throws IOException {
      return knnFormat.fieldsReader(state);
    }

    @Override
    public int getMaxDimensions(String fieldName) {
      return maxDimensions;
    }
  }
}
