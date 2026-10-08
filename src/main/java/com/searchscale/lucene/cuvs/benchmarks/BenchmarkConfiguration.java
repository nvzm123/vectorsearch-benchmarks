package com.searchscale.lucene.cuvs.benchmarks;

import com.nvidia.cuvs.CagraIndexParams.CagraGraphBuildAlgo;
import com.nvidia.cuvs.CagraIndexParams.CodebookGen;
import com.nvidia.cuvs.CagraIndexParams.CudaDataType;
import com.nvidia.cuvs.CagraIndexParams.CuvsDistanceType;
import com.nvidia.cuvs.lucene.AcceleratedHNSWParams;
import com.searchscale.lucene.cuvs.benchmarks.LuceneCuvsBenchmarks.Codex;
import java.util.List;
import org.apache.lucene.index.VectorSimilarityFunction;

public class BenchmarkConfiguration {

  public String benchmarkID;
  public String datasetFile;
  public int indexOfVector;
  public String vectorColName;
  public int numDocs;
  public int vectorDimension;
  public VectorSimilarityFunction vectorSimilarityFunction = VectorSimilarityFunction.EUCLIDEAN;
  public String queryFile;
  public int numQueriesToRun;
  public int numWarmUpQueries;
  public int flushFreq;
  public int topK;
  public int numIndexThreads;
  public int cuvsWriterThreads;
  public int queryThreads;
  public boolean dropPageCacheBeforeSearch = false;
  public boolean prewarmIndexBeforeSearch = true;
  public boolean createIndexInMemory;
  public boolean cleanIndexDirectory;
  public boolean saveResultsOnDisk;
  public String resultsDirectory;
  public boolean hasColNames;
  public Codex algoToRun;
  public String groundTruthFile;
  public String indexDirPath;
  public boolean loadVectorsInMemory;
  public boolean useLuceneDocIdAsVectorId;
  // Chunk size (MB) for the sequential ChunkedVectorProvider used when loadVectorsInMemory=false.
  public int ingestChunkSizeMB = 64;
  public boolean ingestPrefetch =
      true; // double-buffered prefetch reader (numIndexThreads=1 only; falls back otherwise)
  public boolean partitionedOverlap =
      false; // partitioned build: pipeline the K segment builds (ingest overlaps prior GPU commit)
  public int partitionedPipelineDepth = 2; // max co-resident segment builds (peak host = depth*N/K)
  public boolean skipIndexing;
  public int forceMerge;
  public boolean enableTieredMerge;
  public boolean enableIndexWriterInfoStream;
  public boolean useCompoundFile = true;
  public boolean useLargeVectorWriteOutput = false;
  public int largeVectorWriteBufferSizeKB = 1024;
  // Lucene HNSW parameters
  public int hnswMaxConn = 16; // 16 default (max 512)
  public int hnswBeamWidth = 100; // 100 default (max 3200)
  public int hnswMergeThreads;

  // CAGRA parameters
  public int cagraIntermediateGraphDegree; // 128 default
  public int cagraGraphDegree; // 64 default
  public int cagraITopK;
  public int cagraSearchWidth;
  public int cagraHnswLayers; // layers in CAGRA->HNSW conversion
  public List<Integer> efSearch; // e.g. [64] or [64, 128, 256]
  public CagraGraphBuildAlgo cagraGraphBuildAlgo;
  public AcceleratedHNSWParams.Strategy cagraIndexParamsStrategy =
      AcceleratedHNSWParams.Strategy.HEURISTIC;

  // CAGRA IVF_PQ parameters
  public int cuVSIvfPqParamsRefinementRate = 1;
  public boolean cuVSIvfPqIndexParamsAddDataOnBuild = true;
  public CodebookGen cuVSIvfPqIndexParamsCodebookKind = CodebookGen.PER_SUBSPACE;
  public boolean cuVSIvfPqIndexParamsConservativeMemoryAllocation = false;
  public boolean cuVSIvfPqIndexParamsForceRandomRotation = false;
  public int cuVSIvfPqIndexParamsKmeansNIters = 20;
  public double cuVSIvfPqIndexParamsKmeansTrainsetFraction = 0.5;
  public int cuVSIvfPqIndexParamsMaxTrainPointsPerPqCode = 256;
  public CuvsDistanceType cuVSIvfPqIndexParamsMetric = CuvsDistanceType.L2Expanded;
  public float cuVSIvfPqIndexParamsMetricArg = 2.0f;
  public int cuVSIvfPqIndexParamsNLists = 1024;
  public int cuVSIvfPqIndexParamsPqBits = 8;
  public int cuVSIvfPqIndexParamsPqDim = 0;
  public CudaDataType cuVSIvfPqSearchParamsInternalDistanceDtype = CudaDataType.CUDA_R_32F;
  public CudaDataType cuVSIvfPqSearchParamsLutDtype = CudaDataType.CUDA_R_32F;
  public int cuVSIvfPqSearchParamsNProbes = 20;
  public double cuVSIvfPqSearchParamsPreferredShmemCarveout = 1.0;

  // ── IVF-PQ overrides for the forceMerge phase ──────────────────────────────
  /**
   * nLists to use when building the CAGRA graph during {@code forceMerge}.
   *
   * <p>During flush, segments are small so {@link #cuVSIvfPqIndexParamsNLists} is
   * sized accordingly.  A force-merged segment can be 4-8x larger; using the flush
   * value here leaves too few vectors per cluster, degrading graph quality.
   *
   * <p>Per the NVIDIA cuVS documentation, {@code n_rows / n_lists} should fall in
   * the range 1,000–10,000.  Set this to match your merged segment size, e.g.
   * 10 M vectors / 5,000 nLists ≈ 2,000 vectors/cluster.
   *
   * <p>{@code 0} (default) = use {@link #cuVSIvfPqIndexParamsNLists} unchanged.
   */
  public int cuVSIvfPqIndexParamsForceMergeNLists = 0;

  /**
   * nProbes to use when searching the IVF-PQ index during {@code forceMerge}.
   *
   * <p>With a larger nLists at merge time, nProbes should be scaled proportionally
   * to maintain the same recall during CAGRA graph construction.
   *
   * <p>{@code 0} (default) = use {@link #cuVSIvfPqSearchParamsNProbes} unchanged.
   */
  public int cuVSIvfPqSearchParamsForceMergeNProbes = 0;

  public boolean isLucene() {
    return Codex.LUCENE_HNSW.equals(algoToRun);
  }

  public boolean isCagra() {
    return Codex.CAGRA_HNSW.equals(algoToRun);
  }

  public boolean isCagraSearch() {
    return Codex.CAGRA_SEARCH.equals(algoToRun);
  }

  public boolean isCagraHNSWBinary() {
    return Codex.CAGRA_HNSW_BINARY.equals(algoToRun);
  }

  public boolean isCagraHNSWScalar() {
    return Codex.CAGRA_HNSW_SCALAR.equals(algoToRun);
  }

  /**
   * Returns the list of efSearch values to use during search.
   *
   * <p>If {@code efSearch} is set in the config JSON (e.g. [64, 128, 256]),
   * those values are returned directly. Otherwise, falls back to a single-element
   * list containing a default derived from topK.
   *
   * <p>The benchmark runner iterates over these values and runs search once per value
   * against the <b>same</b> index — no rebuild is needed.
   */
  public List<Integer> getEfSearchValues() {
    if (efSearch != null && !efSearch.isEmpty()) {
      return efSearch;
    }
    // Default: 1.5x topK, but at least topK
    return List.of(Math.max(topK, (int) Math.ceil(topK * 1.5)));
  }

  public CuvsDistanceType resolvedCuvsDistanceType() {
    if (vectorSimilarityFunction == null) {
      throw new IllegalArgumentException("vectorSimilarityFunction must not be null");
    }
    return switch (vectorSimilarityFunction) {
      case EUCLIDEAN -> CuvsDistanceType.L2Expanded;
      case DOT_PRODUCT, MAXIMUM_INNER_PRODUCT -> CuvsDistanceType.InnerProduct;
      case COSINE -> CuvsDistanceType.CosineExpanded;
    };
  }

  public String prettyString() {
    StringBuilder sb = new StringBuilder();
    sb.append("Benchmark ID: ").append(benchmarkID).append('\n');
    sb.append("Dataset file used is: ").append(datasetFile).append('\n');
    sb.append("Index of vector field is: ").append(indexOfVector).append('\n');
    sb.append("Name of the vector field is: ").append(vectorColName).append('\n');
    sb.append("Number of documents to be indexed are: ").append(numDocs).append('\n');
    sb.append("Number of dimensions are: ").append(vectorDimension).append('\n');
    sb.append("Vector similarity function: ").append(vectorSimilarityFunction).append('\n');
    sb.append("Resolved cuVS distance type: ").append(resolvedCuvsDistanceType()).append('\n');
    sb.append("Query file used is: ").append(queryFile).append('\n');
    sb.append("Number of queries to run: ").append(numQueriesToRun).append('\n');
    sb.append("Number of warmup queries: ").append(numWarmUpQueries).append('\n');
    sb.append("Flush frequency (every n documents): ").append(flushFreq).append('\n');
    sb.append("TopK value is: ").append(topK).append('\n');
    sb.append("numIndexThreads is: ").append(numIndexThreads).append('\n');
    sb.append("Query threads: ").append(queryThreads).append('\n');
    sb.append("Drop page cache before search: ").append(dropPageCacheBeforeSearch).append('\n');
    sb.append("Prewarm index before search: ").append(prewarmIndexBeforeSearch).append('\n');
    sb.append("Create index in memory: ").append(createIndexInMemory).append('\n');
    sb.append("Clean index directory: ").append(cleanIndexDirectory).append('\n');
    sb.append("Save results on disk: ").append(saveResultsOnDisk).append('\n');
    sb.append("Has column names in the dataset file: ").append(hasColNames).append('\n');
    sb.append("algoToRun: ").append(algoToRun).append('\n');
    sb.append("Ground Truth file used is: ").append(groundTruthFile).append('\n');
    sb.append("index directory path is: ").append(indexDirPath).append('\n');
    sb.append("Load vectors in memory before indexing: ").append(loadVectorsInMemory).append('\n');
    sb.append("Use Lucene doc ID as vector ID: ").append(useLuceneDocIdAsVectorId).append('\n');
    sb.append("Ingest chunk size (MB): ").append(ingestChunkSizeMB).append('\n');
    sb.append("Skip indexing (and use existing index for search): ")
        .append(skipIndexing)
        .append('\n');
    sb.append("Do force merge while indexing documents [a value < 1 implies no force merge]: ")
        .append(forceMerge)
        .append('\n');
    sb.append("Enable TieredMerge: ").append(enableTieredMerge).append('\n');
    sb.append("Use compound file: ").append(useCompoundFile).append('\n');
    sb.append("Use large vector write output: ").append(useLargeVectorWriteOutput).append('\n');
    sb.append("Large vector write buffer (KiB): ")
        .append(largeVectorWriteBufferSizeKB)
        .append('\n');
    sb.append("Num HNSW merge threads: ").append(hnswMergeThreads).append('\n');
    sb.append("enableIndexWriterInfoStream: ").append(enableIndexWriterInfoStream).append('\n');
    sb.append("efSearch: ").append(getEfSearchValues()).append('\n');
    sb.append("nLists (flush):            ").append(cuVSIvfPqIndexParamsNLists).append('\n');
    sb.append("nLists (forceMerge):       ")
        .append(cuVSIvfPqIndexParamsForceMergeNLists)
        .append(cuVSIvfPqIndexParamsForceMergeNLists == 0 ? "  (same as flush)" : "")
        .append('\n');
    sb.append("nProbes (flush):           ").append(cuVSIvfPqSearchParamsNProbes).append('\n');
    sb.append("nProbes (forceMerge):      ")
        .append(cuVSIvfPqSearchParamsForceMergeNProbes)
        .append(cuVSIvfPqSearchParamsForceMergeNProbes == 0 ? "  (same as flush)" : "")
        .append('\n');

    sb.append("------- algo parameters ------\n");
    if (isLucene()) {
      sb.append("hnswMaxConn: ").append(hnswMaxConn).append('\n');
      sb.append("hnswBeamWidth: ").append(hnswBeamWidth).append('\n');
    } else {
      sb.append("cagraIntermediateGraphDegree: ").append(cagraIntermediateGraphDegree).append('\n');
      sb.append("cagraGraphDegree: ").append(cagraGraphDegree).append('\n');
      sb.append("cuvsWriterThreads: ").append(cuvsWriterThreads).append('\n');
      sb.append("cagraITopK: ").append(cagraITopK).append('\n');
      sb.append("cagraSearchWidth: ").append(cagraSearchWidth).append('\n');
      sb.append("cagraHnswLayers: ").append(cagraHnswLayers).append('\n');
      sb.append("cagraGraphBuildAlgo: ").append(cagraGraphBuildAlgo).append('\n');
      sb.append("cagraIndexParamsStrategy: ").append(cagraIndexParamsStrategy).append('\n');
    }
    return sb.toString();
  }

  @Override
  public String toString() {
    return prettyString();
  }

  public void debugPrintArguments() {
    // keep a single source of truth for printing
    System.out.print(prettyString());
  }
}
