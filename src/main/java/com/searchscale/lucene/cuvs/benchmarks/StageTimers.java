/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.searchscale.lucene.cuvs.benchmarks;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.LongAdder;

/**
 * Lightweight, thread-safe accumulator for coarse-grained stage timings used to profile the
 * index-build pipeline from the benchmark harness. Accumulated stage totals are printed once at JVM
 * shutdown.
 *
 * <p>Stages are tagged with the dominant resource they exercise ([CPU], [DISK], [GPU]). Stages that
 * move a known number of bytes also report effective throughput (MB/s). Diagnostic-only scaffolding:
 * disable with {@code -Dcuvs.stageTimers=false} or remove wholesale. The cuvs-lucene writer prints a
 * complementary breakdown of the in-writer sub-stages.
 */
public final class StageTimers {

  private static final Map<String, LongAdder> NANOS = new ConcurrentHashMap<>();
  private static final Map<String, LongAdder> COUNTS = new ConcurrentHashMap<>();
  private static final Map<String, LongAdder> BYTES = new ConcurrentHashMap<>();
  private static final boolean ENABLED =
      !"false".equalsIgnoreCase(System.getProperty("cuvs.stageTimers", "true"));

  static {
    if (ENABLED) {
      Runtime.getRuntime()
          .addShutdownHook(
              new Thread(() -> dump("benchmark stage timers"), "benchmark-stage-timers-dump"));
    }
  }

  private StageTimers() {}

  /** Returns a start timestamp (nanoseconds) to later pass to {@link #stop}. */
  public static long start() {
    return System.nanoTime();
  }

  /** Accumulates elapsed time since {@code startNanos} against the named stage. */
  public static void stop(String stage, long startNanos) {
    record(stage, System.nanoTime() - startNanos, 0L);
  }

  /** Accumulates elapsed time and the number of bytes moved by the stage (for throughput). */
  public static void stop(String stage, long startNanos, long bytes) {
    record(stage, System.nanoTime() - startNanos, bytes);
  }

  /** Directly accumulates a precomputed duration (nanoseconds) against the named stage. */
  public static void record(String stage, long nanos) {
    record(stage, nanos, 0L);
  }

  /** Directly accumulates a duration and the number of bytes moved by the stage. */
  public static void record(String stage, long nanos, long bytes) {
    if (!ENABLED) {
      return;
    }
    NANOS.computeIfAbsent(stage, k -> new LongAdder()).add(nanos);
    COUNTS.computeIfAbsent(stage, k -> new LongAdder()).increment();
    if (bytes > 0) {
      BYTES.computeIfAbsent(stage, k -> new LongAdder()).add(bytes);
    }
  }

  /** Appends a stable, machine-readable snapshot to a benchmark result map. */
  public static void appendMetrics(Map<String, Object> target, String prefix) {
    NANOS.keySet().stream()
        .sorted()
        .forEach(
            stage -> {
              String key = prefix + "/" + stage;
              target.put(key + "/seconds", NANOS.get(stage).sum() / 1e9);
              target.put(key + "/count", COUNTS.getOrDefault(stage, new LongAdder()).sum());
              LongAdder bytes = BYTES.get(stage);
              if (bytes != null) {
                target.put(key + "/bytes", bytes.sum());
              }
            });
  }

  /** Prints accumulated stage totals (with throughput where bytes are known), sorted by time. */
  public static void dump(String header) {
    if (NANOS.isEmpty()) {
      return;
    }
    StringBuilder sb = new StringBuilder();
    sb.append(System.lineSeparator()).append("===== ").append(header).append(" =====");
    sb.append(System.lineSeparator());
    NANOS.entrySet().stream()
        .sorted((a, b) -> Long.compare(b.getValue().sum(), a.getValue().sum()))
        .forEach(
            e -> {
              String stage = e.getKey();
              double secs = e.getValue().sum() / 1e9;
              long count = COUNTS.getOrDefault(stage, new LongAdder()).sum();
              LongAdder b = BYTES.get(stage);
              if (b != null && b.sum() > 0 && secs > 0) {
                double mbps = (b.sum() / 1e6) / secs;
                double gb = b.sum() / 1e9;
                sb.append(
                    String.format(
                        "  %-42s %10.3f s   (n=%d)  %9.1f MB/s  (%.2f GB)%n",
                        stage, secs, count, mbps, gb));
              } else {
                sb.append(String.format("  %-42s %10.3f s   (n=%d)%n", stage, secs, count));
              }
            });
    sb.append("=================================================");
    sb.append(System.lineSeparator());
    System.err.print(sb);
    System.err.flush();
  }
}
