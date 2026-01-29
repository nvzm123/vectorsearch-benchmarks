package com.searchscale.lucene.cuvs.benchmarks;

import static org.apache.lucene.index.VectorSimilarityFunction.EUCLIDEAN;

import java.io.File;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.Charset;
import java.nio.file.FileVisitOption;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import org.apache.commons.io.FileUtils;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.codecs.lucene101.Lucene101Codec;
import org.apache.lucene.codecs.lucene101.Lucene101Codec.Mode;
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

import com.nvidia.cuvs.lucene.GPUKnnFloatVectorQuery;
import com.nvidia.cuvs.lucene.Lucene99AcceleratedHNSWVectorsFormat;

public class LuceneCuvsBenchmarks {

  private static final Logger log = LoggerFactory.getLogger(LuceneCuvsBenchmarks.class.getName());

  private static boolean RESULTS_DEBUGGING = false; // when enabled, titles are indexed and printed after search
  private static boolean INDEX_WRITER_INFO_STREAM = true; // when enabled, prints information about merges, deletes,

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

  @SuppressWarnings("resource")
  public static void main(String[] args) throws Throwable {

    if (args.length < 1 || args.length > 3) {
      System.err.println("Usage: ./benchmarks.sh <jobs-file> [benchmarkID] [resultsDir]");
      return;
    }

    BenchmarkConfiguration config = Util.newObjectMapper().readValue(new File(args[0]), BenchmarkConfiguration.class);
    
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
        log.info("Time taken to load {} vectors in-memory: {} ms", loadedVectors.size(), (System.currentTimeMillis() - start));
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
        log.info("Mapdb loaded. Now loading all vectors in memory (loadVectorsInMemory is enabled)");
        long start = System.currentTimeMillis();
        List<float[]> loadedVectors = new ArrayList<float[]>(vectors.size());
        for (int i = 0; i < vectors.size(); i++) {
          loadedVectors.add(vectors.get(i));
        }
        vectorProvider = new MemoryVectorProvider(loadedVectors);
        db.close();
        log.info("Time taken to load the vectors in-memory is: {}", (System.currentTimeMillis() - start));
      } else {
        vectorProvider = new MapDBVectorProvider(vectors, db);
      }
    }

    try {

      log.info("Time taken for parsing/loading dataset is {} ms", (System.currentTimeMillis() - parseStartTime));

      // [2] Benchmarking setup

      // HNSW Writer:
      IndexWriterConfig luceneHNSWWriterConfig = new IndexWriterConfig(new StandardAnalyzer());
      luceneHNSWWriterConfig.setCodec(getLuceneHnswCodec(config));
      //luceneHNSWWriterConfig.setUseCompoundFile(false);
      // Configure to flush based on document count only
      // For 4M docs with 768-dim float vectors, we need approximately:
      // 4M * 768 * 4 bytes = ~12GB just for vectors, plus overhead
      // Set RAM buffer to 32GB to ensure doc count triggers flush first
      luceneHNSWWriterConfig.setRAMBufferSizeMB(32768); // 32GB RAM buffer
      luceneHNSWWriterConfig.setMaxBufferedDocs(config.flushFreq);
      luceneHNSWWriterConfig.setMergePolicy(NoMergePolicy.INSTANCE);
      // Use reflection to bypass the 2048MB per-thread limit and set it to 10GB
      setPerThreadRAMLimit(luceneHNSWWriterConfig, 10240); // 10GB per thread
      log.info("Configured HNSW writer - MaxBufferedDocs: {}, RAMBufferSizeMB: {}, PerThreadRAMLimit: {} MB", 
              config.flushFreq, luceneHNSWWriterConfig.getRAMBufferSizeMB(), 
              luceneHNSWWriterConfig.getRAMPerThreadHardLimitMB());

      IndexWriterConfig cuvsIndexWriterConfig = new IndexWriterConfig(new StandardAnalyzer());
      cuvsIndexWriterConfig.setCodec(getCuVSCodec(config));
      //cuvsIndexWriterConfig.setUseCompoundFile(false);
      // Configure to flush based on document count only
      // For 4M docs with 768-dim float vectors, we need approximately:
      // 4M * 768 * 4 bytes = ~12GB just for vectors, plus overhead
      // Set RAM buffer to 32GB to ensure doc count triggers flush first
      cuvsIndexWriterConfig.setRAMBufferSizeMB(32768); // 32GB RAM buffer
      cuvsIndexWriterConfig.setMaxBufferedDocs(config.flushFreq);
      cuvsIndexWriterConfig.setMergePolicy(NoMergePolicy.INSTANCE);
      // Use reflection to bypass the 2048MB per-thread limit and set it to 10GB
      setPerThreadRAMLimit(cuvsIndexWriterConfig, 10240); // 10GB per thread
      log.info("Configured CuVS writer - MaxBufferedDocs: {}, RAMBufferSizeMB: {}, PerThreadRAMLimit: {} MB", 
              config.flushFreq, cuvsIndexWriterConfig.getRAMBufferSizeMB(),
              cuvsIndexWriterConfig.getRAMPerThreadHardLimitMB());

      if (INDEX_WRITER_INFO_STREAM) {
        luceneHNSWWriterConfig.setInfoStream(new PrintStreamInfoStream(System.out));
        cuvsIndexWriterConfig.setInfoStream(new PrintStreamInfoStream(System.out));
      }

     	if (!config.skipIndexing) {


      IndexWriter luceneHnswIndexWriter = null;
      IndexWriter cuvsIndexWriter = null;

      
      
      if (config.algoToRun.equalsIgnoreCase("LUCENE_HNSW")) {
        if (!config.createIndexInMemory) {
          Path hnswIndex = Path.of(config.hnswIndexDirPath);
          luceneHnswIndexWriter = new IndexWriter(FSDirectory.open(hnswIndex), luceneHNSWWriterConfig);
        } else {
          luceneHnswIndexWriter = new IndexWriter(new ByteBuffersDirectory(), luceneHNSWWriterConfig);
        }
      } else if (config.algoToRun.equalsIgnoreCase("CAGRA_HNSW")) {
        if (!config.createIndexInMemory) {
          Path cuvsIndex = Path.of(config.cuvsIndexDirPath);
          cuvsIndexWriter = new IndexWriter(FSDirectory.open(cuvsIndex), cuvsIndexWriterConfig);
        } else {
          cuvsIndexWriter = new IndexWriter(new ByteBuffersDirectory(), cuvsIndexWriterConfig);
        }
      }


      IndexWriter writer;

      if ("LUCENE_HNSW".equalsIgnoreCase(config.algoToRun)) {
        writer = luceneHnswIndexWriter;
      } else if ("CAGRA_HNSW".equalsIgnoreCase(config.algoToRun)) {
        writer = cuvsIndexWriter;
      } else {
        throw new IllegalArgumentException("Please pass an acceptable option for `algoToRun`. Choices: LUCENE_HNSW, CAGRA_HNSW");
      }

        var formatName = writer.getConfig().getCodec().knnVectorsFormat().getName();
   	  
        boolean isCuVSIndexing = formatName.equals("Lucene99AcceleratedHNSWVectorsFormat");

        log.info("Indexing documents using {} ...", formatName);
        long indexStartTime = System.currentTimeMillis();
        indexDocuments(writer, config, titles, vectorProvider);
        long indexTimeTaken = System.currentTimeMillis() - indexStartTime;
        if (isCuVSIndexing) {
          metrics.put("cuvs-indexing-time", indexTimeTaken);
        } else {
          metrics.put("hnsw-indexing-time", indexTimeTaken);
        }

        log.info("Time taken for index building (end to end): {} ms", indexTimeTaken);

        boolean usingFSDirectory = luceneHnswIndexWriter != null
            ? luceneHnswIndexWriter.getDirectory() instanceof FSDirectory
            : cuvsIndexWriter.getDirectory() instanceof FSDirectory;

        try {
          if (usingFSDirectory) {
            Path indexPath = writer == cuvsIndexWriter ? Paths.get(config.cuvsIndexDirPath)
                : Paths.get(config.hnswIndexDirPath);
            long directorySize;
            try (var stream = Files.walk(indexPath, FileVisitOption.FOLLOW_LINKS)) {
              directorySize = stream.filter(p -> p.toFile().isFile()).mapToLong(p -> p.toFile().length()).sum();
            }
            double directorySizeGB = directorySize / 1_073_741_824.0;
            if (writer == cuvsIndexWriter) {
              metrics.put("cuvs-index-size", directorySizeGB);
            } else {
              metrics.put("hnsw-index-size", directorySizeGB);
            }
            log.info("Size of {}: {} GB", indexPath.toString(), directorySizeGB);
          }
        } catch (IOException e) {
          log.error("Failed to calculate directory size for {}",
              writer == cuvsIndexWriter ? config.cuvsIndexDirPath : config.hnswIndexDirPath, e);
        }
       }
     	
      Directory indexDir = MMapDirectory.open("CAGRA_HNSW".equals(config.algoToRun) ? Path.of(config.cuvsIndexDirPath) : Path.of(config.hnswIndexDirPath));
      log.info("Index directory is: {} (using memory-mapped files)", indexDir);
      log.info("Querying documents using {} ...", config.algoToRun);
      // Always use standard Lucene search since we always create Lucene HNSW indexes
      search(indexDir, config, false, metrics, queryResults,
        Util.readGroundTruthFile(config.groundTruthFile));

      Util.calculateRecallAccuracy(queryResults, metrics, "CAGRA_HNSW".equalsIgnoreCase(config.algoToRun));

      String resultsJson = Util.newObjectMapper().writerWithDefaultPrettyPrinter()
          .writeValueAsString(Map.of("configuration", config, "metrics", metrics));

      if (config.saveResultsOnDisk) {
        // Use the resultsDirectory directly if provided
        String resultsDir = config.resultsDirectory != null ? config.resultsDirectory : "results";
        File results = new File(resultsDir);
        if (!results.exists()) {
          results.mkdirs();
        }

        // Save results.json directly to the specified directory
        FileUtils.write(
            new File(results.toString() + "/results.json"),
            resultsJson, Charset.forName("UTF-8"));
            
        // Save CSV with neighbors data  
        Util.writeCSV(queryResults, results.toString() + "/neighbors.csv");
        
        log.info("Results saved to directory: {}", resultsDir);
      }

      log.info("\n-----\nOverall metrics: " + metrics + "\nMetrics: \n" + resultsJson + "\n-----");
      
      // Close the index directory before cleaning
      indexDir.close();
      
      // Clean index directory after benchmarks complete if requested
      if (config.cleanIndexDirectory && !config.createIndexInMemory) {
        Path indexPath = null;
        if (config.algoToRun.equalsIgnoreCase("LUCENE_HNSW")) {
          indexPath = Path.of(config.hnswIndexDirPath);
        } else if (config.algoToRun.equalsIgnoreCase("CAGRA_HNSW")) {
          indexPath = Path.of(config.cuvsIndexDirPath);
        }
        
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
  }

  private static void indexDocuments(IndexWriter writer, BenchmarkConfiguration config, List<String> titles,
      VectorProvider vectorProvider) throws IOException, InterruptedException {

    int threads = config.numIndexThreads;
    ExecutorService pool = Executors.newFixedThreadPool(threads);
    AtomicInteger numDocsIndexed = new AtomicInteger(0);
    log.info("Starting indexing with {} threads.", threads);
    log.info("IndexWriter config - MaxBufferedDocs: {}, RAMBufferSizeMB: {}", 
            writer.getConfig().getMaxBufferedDocs(), writer.getConfig().getRAMBufferSizeMB());
    final int numDocsToIndex = Math.min(config.numDocs, vectorProvider.size());

    for (int i = 0; i < threads; i++) {
      pool.submit(() -> {
        int localCount = 0;
        while (true) {
          int id = numDocsIndexed.getAndIncrement();
          if (id >= numDocsToIndex) {
            break; // done
          }
          float[] vector;
          try {
            vector = Objects.requireNonNull(vectorProvider.get(id));
            localCount++;
          } catch (IOException e) {
            throw new UncheckedIOException("Failed to read vector at index " + id, e);
          }
          Document doc = new Document();
          doc.add(new StringField("id", String.valueOf(id), Field.Store.YES));
          doc.add(new KnnFloatVectorField(config.vectorColName, vector, EUCLIDEAN));
          if (RESULTS_DEBUGGING)
            doc.add(new StringField("title", titles.get(id), Field.Store.YES));
          try {
            writer.addDocument(doc);
            if ((id + 1) % 25000 == 0) {
              log.info("Done indexing {} documents. Pending docs: {}", (id + 1), writer.getPendingNumDocs());
            }
            // Log when we expect a flush
            if ((id + 1) == config.flushFreq || (id + 1) == 2 * config.flushFreq) {
              log.info("Expected flush point reached at {} documents", (id + 1));
            }
          } catch (IOException ex) {
            throw new UncheckedIOException(ex);
          }
        }
      });
    }
    pool.shutdown();
    pool.awaitTermination(Long.MAX_VALUE, TimeUnit.SECONDS);

    // log.info("Calling forceMerge(1).");
    // writer.forceMerge(1);
    log.info("Calling commit.");
    writer.commit();
    writer.close();
  }

  private static void search(Directory directory, BenchmarkConfiguration config, boolean useCuVS,
      Map<String, Object> metrics, List<QueryResult> queryResults, List<int[]> groundTruth) {
	  
    DB db = null;
    try (IndexReader indexReader = DirectoryReader.open(directory)) {
      IndexSearcher indexSearcher = new IndexSearcher(indexReader);

      IndexTreeList<float[]> queries;
      String queryMapdbFile = config.queryFile + ".mapdb";

      if (new File(queryMapdbFile).exists() == false) {
        log.info("No mapdb file found for queries. Reading source files to build one ...");
        db = DBMaker.fileDB(queryMapdbFile).make();
        queries = db.indexTreeList("vectors", SERIALIZER.FLOAT_ARRAY).createOrOpen();

              if (config.queryFile.endsWith(".csv")) {
        for (String line : FileUtils.readFileToString(new File(config.queryFile), "UTF-8").split("\n")) {
          queries.add(Util.parseFloatArrayFromStringArray(line));
        }
      } else if (config.queryFile.contains("fvecs")) {
        FBIvecsReader.readFvecs(config.queryFile, -1, queries);
      } else if (config.queryFile.contains("fbin")) {
        FBIvecsReader.readFbin(config.queryFile, -1, queries);
      } else if (config.queryFile.contains("bvecs")) {
        FBIvecsReader.readBvecs(config.queryFile, -1, queries);
        }
        log.info("Mapdb file created with {} number of queries", queries.size());
      } else {
        log.info("Mapdb file found for queries. Loading ...");
        db = DBMaker.fileDB(queryMapdbFile).make();
        queries = db.indexTreeList("vectors", SERIALIZER.FLOAT_ARRAY).createOrOpen();
        log.info("{} queries available from the mapdb file", queries.size());
      }

      int qThreads = config.queryThreads;
      if (useCuVS)
        qThreads = 1;
      ExecutorService pool = Executors.newFixedThreadPool(qThreads);
      AtomicInteger queriesFinished = new AtomicInteger(0);
      ConcurrentHashMap<Integer, Double> queryLatencies = new ConcurrentHashMap<Integer, Double>();
      ConcurrentHashMap<Integer, Double> retrievalLatencies = new ConcurrentHashMap<Integer, Double>();

      long startTime = System.currentTimeMillis();
      AtomicInteger queryId = new AtomicInteger(0);
      queries.stream().limit(config.numQueriesToRun).forEach((queryVector) -> {
        // Get a unique query ID for this query before submitting to thread pool
        int currentQueryId = queryId.getAndIncrement();
        pool.submit(() -> {
          KnnFloatVectorQuery query;

          if (useCuVS) {
            int effectiveEfSearch = config.getEffectiveEfSearch();
            query = new GPUKnnFloatVectorQuery(config.vectorColName, queryVector, effectiveEfSearch, null, config.cagraITopK,
                                               config.cagraSearchWidth);
          } else {
              int effectiveEfSearch = config.getEffectiveEfSearch();
              query = new KnnFloatVectorQuery(config.vectorColName, queryVector, effectiveEfSearch);   
          }

          TopDocs topDocs;
          long searchStartTime = System.nanoTime();
          try {
                int effectiveEfSearch = config.getEffectiveEfSearch();
                TopScoreDocCollectorManager collectorManager = new TopScoreDocCollectorManager(effectiveEfSearch, null,
                Integer.MAX_VALUE, true);
            topDocs = indexSearcher.search(query, collectorManager);
          } catch (IOException e) {
            throw new RuntimeException("Problem during executing a query: ", e);
          }
          double searchTimeTakenMs = TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - searchStartTime);
          // log.info("End to end search took: " + searchTimeTakenMs);
          if (currentQueryId > config.numWarmUpQueries) {
        	  queryLatencies.put(queryId.get(), searchTimeTakenMs);
          }
          int finishedCount = queriesFinished.incrementAndGet();

          // Log progress every 2 queries
          if (finishedCount % 2 == 0 || finishedCount == config.numQueriesToRun) {
            log.info("Done querying " + finishedCount + " out of " + config.numQueriesToRun + " queries.");
          }

          ScoreDoc[] hits = topDocs.scoreDocs;
          List<Integer> neighbors = new ArrayList<>();
          List<Float> scores = new ArrayList<>();

          // Debug: Log search results for first query
          if (queryId.get() == 0) {
            log.info("Debug: First query returned " + hits.length + " hits (ef-search candidates)");
            log.info("Debug: Will select top " + config.topK + " from " + hits.length + " candidates");
          }
          int numResultsToTake = Math.min(config.topK, hits.length);
          long retrievalStartTime = System.nanoTime();
          for (int i = 0; i < numResultsToTake; i++) {
            ScoreDoc hit = hits[i];
            try {
              Document d = indexReader.storedFields().document(hit.doc);
              neighbors.add(Integer.parseInt(d.get("id")));
            } catch (IOException e) {
              e.printStackTrace();
            }
            scores.add(hit.score);
          }
          double retrievalTimeTakenMs = TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - retrievalStartTime);
          if (currentQueryId > config.numWarmUpQueries) {
        	  retrievalLatencies.put(queryId.get(), retrievalTimeTakenMs);
          }          
          
          // Debug: Log results for all queries
          log.info("Query " + currentQueryId + " - First 5 neighbors: " + neighbors.subList(0, Math.min(5, neighbors.size())));
          log.info("Query " + currentQueryId + " - First 5 distances: " + scores.subList(0, Math.min(5, scores.size())));
          int[] expectedNeighbors = groundTruth.get(currentQueryId);
          log.info("Query " + currentQueryId + " - Expected neighbors: " + java.util.Arrays.toString(java.util.Arrays.copyOf(expectedNeighbors, Math.min(5, expectedNeighbors.length))));

          var s = useCuVS ? "lucene_cuvs" : "lucene_hnsw";
          if (currentQueryId > config.numWarmUpQueries) {
	          QueryResult result = new QueryResult(s, currentQueryId, neighbors, groundTruth.get(currentQueryId), scores,
	              searchTimeTakenMs);          
	          queryResults.add(result);
          } else {
        	  log.info("Skipping warmup query: {}", currentQueryId);
          }
        });
      });

      pool.shutdown();
      pool.awaitTermination(Long.MAX_VALUE, TimeUnit.SECONDS);

      long endTime = System.currentTimeMillis();

      metrics.put((useCuVS ? "cuvs" : "hnsw") + "-query-time", (endTime - startTime));
      metrics.put((useCuVS ? "cuvs" : "hnsw") + "-query-throughput",
          (queryLatencies.size() / ((endTime - startTime) / 1000.0)));
      double avgLatency = new ArrayList<>(queryLatencies.values()).stream().reduce(0.0, Double::sum)
          / queryLatencies.size();
      double avgRetLatency = new ArrayList<>(retrievalLatencies.values()).stream().reduce(0.0, Double::sum)
              / retrievalLatencies.size();

      metrics.put((useCuVS ? "cuvs" : "hnsw") + "-mean-latency", avgLatency);
      metrics.put((useCuVS ? "cuvs" : "hnsw") + "-mean-retrieval-latency", avgRetLatency);

      // Add segment count to metrics
      int segmentCount = indexReader.leaves().size();
      metrics.put((useCuVS ? "cuvs" : "hnsw") + "-segment-count", segmentCount);

    } catch (Exception e) {
      e.printStackTrace();
      log.error("Exception during querying", e);
    } finally {
      if (db != null) {
        db.close();
      }
    }
  }

  private static Lucene101Codec getLuceneHnswCodec(BenchmarkConfiguration config) {
    return new Lucene101Codec(Mode.BEST_SPEED) {

      @Override
      public KnnVectorsFormat getKnnVectorsFormatForField(String field) {
        KnnVectorsFormat knnFormat = new Lucene99HnswVectorsFormat(config.hnswMaxConn, config.hnswBeamWidth);
        // KnnVectorsFormat knnFormat = new Lucene99HnswVectorsFormat(DEFAULT_MAX_CONN,
        // DEFAULT_BEAM_WIDTH);
        return new HighDimensionKnnVectorsFormat(knnFormat, config.vectorDimension);
      }
    };
  }

  private static Codec getCuVSCodec(BenchmarkConfiguration config) {
    // Use Lucene99AcceleratedHNSWVectorsFormat with configurable parameters
    // Constructor signature: (cuvsWriterThreads, intGraphDegree, graphDegree, hnswLayers, maxConn, beamWidth)
    return new Lucene101Codec(Mode.BEST_SPEED) {
      @Override
      public KnnVectorsFormat getKnnVectorsFormatForField(String field) {
        return new Lucene99AcceleratedHNSWVectorsFormat(
            config.cuvsWriterThreads,
            config.cagraIntermediateGraphDegree,
            config.cagraGraphDegree,
            config.cagraHnswLayers,
            config.hnswMaxConn,
            config.hnswBeamWidth);
      }
    };
  }

  // Removed ConfigurableCuVSCodec - using CuVSCPUSearchCodec directly with better error handling

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
