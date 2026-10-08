package com.searchscale.lucene.cuvs.benchmarks;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import com.nvidia.cuvs.CagraIndexParams.CagraGraphBuildAlgo;
import com.nvidia.cuvs.CagraIndexParams.CuvsDistanceType;
import com.nvidia.cuvs.CuVSIvfPqParams;
import com.nvidia.cuvs.lucene.AcceleratedHNSWParams;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.apache.lucene.document.Document;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

public class AppTest {

  @Rule public final TemporaryFolder temporaryFolder = new TemporaryFolder();

  @Test
  public void resolvesLuceneSimilarityToCuvsMetric() {
    BenchmarkConfiguration config = new BenchmarkConfiguration();
    config.vectorSimilarityFunction = VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT;

    assertEquals(CuvsDistanceType.InnerProduct, config.resolvedCuvsDistanceType());
  }

  @Test
  public void rejectsMismatchedCustomIvfPqMetric() {
    BenchmarkConfiguration config = customIvfPqConfig();
    config.vectorSimilarityFunction = VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT;
    config.cuVSIvfPqIndexParamsMetric = CuvsDistanceType.L2Expanded;

    assertThrows(
        IllegalArgumentException.class,
        () -> LuceneCuvsBenchmarks.validateMetricConfiguration(config));
  }

  @Test
  public void buildsExplicitIvfPqParameters() {
    BenchmarkConfiguration config = customIvfPqConfig();
    config.cuVSIvfPqIndexParamsPqDim = 96;
    config.cuVSIvfPqIndexParamsNLists = 1000;
    config.cuVSIvfPqSearchParamsNProbes = 8;

    CuVSIvfPqParams params = LuceneCuvsBenchmarks.buildCustomIvfPqParams(config, 1_000_000);

    assertEquals(96, params.getIndexParams().getPqDim());
    assertEquals(1000, params.getIndexParams().getnLists());
    assertEquals(8, params.getSearchParams().getnProbes());
  }

  @Test
  public void includesExternalIdByDefault() {
    BenchmarkConfiguration config = vectorConfig();

    Document document = LuceneCuvsBenchmarks.createDocument(config, 7, new float[] {1, 2});

    assertNotNull(document.getField("id"));
    assertNotNull(document.getField("vector"));
  }

  @Test
  public void omitsExternalIdWhenLuceneDocIdIsTheVectorId() {
    BenchmarkConfiguration config = vectorConfig();
    config.useLuceneDocIdAsVectorId = true;

    Document document = LuceneCuvsBenchmarks.createDocument(config, 7, new float[] {1, 2});

    assertNull(document.getField("id"));
    assertNotNull(document.getField("vector"));
  }

  @Test
  public void resolvesLuceneDocIdAsVectorIdWithoutAStoredId() {
    BenchmarkConfiguration config = vectorConfig();
    config.useLuceneDocIdAsVectorId = true;

    assertEquals(17, LuceneCuvsBenchmarks.resolveVectorId(config, 17, null));
  }

  @Test
  public void resolvesStoredExternalIdByDefault() {
    BenchmarkConfiguration config = vectorConfig();

    assertEquals(23, LuceneCuvsBenchmarks.resolveVectorId(config, 17, "23"));
  }

  @Test
  public void initializesGpuOnlyWhenTheSelectedPhaseUsesIt() {
    BenchmarkConfiguration config = new BenchmarkConfiguration();
    config.algoToRun = LuceneCuvsBenchmarks.Codex.CAGRA_HNSW_SCALAR;

    assertTrue(LuceneCuvsBenchmarks.requiresGpuInitialization(config));

    config.skipIndexing = true;
    assertFalse(LuceneCuvsBenchmarks.requiresGpuInitialization(config));

    config.algoToRun = LuceneCuvsBenchmarks.Codex.CAGRA_SEARCH;
    assertTrue(LuceneCuvsBenchmarks.requiresGpuInitialization(config));

    config.algoToRun = LuceneCuvsBenchmarks.Codex.LUCENE_HNSW;
    assertFalse(LuceneCuvsBenchmarks.requiresGpuInitialization(config));
  }

  @Test
  public void readsTheNamedFbinInsteadOfItsSiblingCache() throws Exception {
    Path queryFile = temporaryFolder.newFile("queries.fbin").toPath();
    ByteBuffer contents = ByteBuffer.allocate(16).order(ByteOrder.LITTLE_ENDIAN);
    contents.putInt(1).putInt(2).putFloat(1.25f).putFloat(-3.5f);
    Files.write(queryFile, contents.array());
    Files.writeString(Path.of(queryFile + ".mapdb"), "stale cache must not be opened");

    assertTrue(LuceneCuvsBenchmarks.isExplicitBinaryQueryFile(queryFile.toString()));
    List<float[]> queries =
        LuceneCuvsBenchmarks.readExplicitBinaryQueries(queryFile.toString(), 1);

    assertEquals(1, queries.size());
    assertArrayEquals(new float[] {1.25f, -3.5f}, queries.get(0), 0.0f);
  }

  private static BenchmarkConfiguration customIvfPqConfig() {
    BenchmarkConfiguration config = new BenchmarkConfiguration();
    config.cagraIndexParamsStrategy = AcceleratedHNSWParams.Strategy.CUSTOM;
    config.cagraGraphBuildAlgo = CagraGraphBuildAlgo.IVF_PQ;
    config.vectorSimilarityFunction = VectorSimilarityFunction.EUCLIDEAN;
    config.cuVSIvfPqIndexParamsMetric = CuvsDistanceType.L2Expanded;
    return config;
  }

  private static BenchmarkConfiguration vectorConfig() {
    BenchmarkConfiguration config = new BenchmarkConfiguration();
    config.vectorColName = "vector";
    config.vectorSimilarityFunction = VectorSimilarityFunction.EUCLIDEAN;
    return config;
  }
}
