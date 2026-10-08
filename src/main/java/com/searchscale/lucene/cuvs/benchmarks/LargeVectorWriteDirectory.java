/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.searchscale.lucene.cuvs.benchmarks;

import java.io.IOException;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.StandardOpenOption;
import java.util.Arrays;
import java.util.Set;
import java.util.stream.Collectors;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.FilterDirectory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.store.OutputStreamIndexOutput;

/** Uses a configurable output buffer for Lucene {@code .vec} files. */
public final class LargeVectorWriteDirectory extends FilterDirectory {

  private final FSDirectory fsDirectory;
  private final int bufferBytes;
  private final Set<String> bufferedExtensions;

  public LargeVectorWriteDirectory(FSDirectory in, int bufferBytes) {
    this(in, bufferBytes, configuredExtensions());
  }

  LargeVectorWriteDirectory(
      FSDirectory in, int bufferBytes, Set<String> bufferedExtensions) {
    super(in);
    if (bufferBytes < 8) {
      throw new IllegalArgumentException("bufferBytes must be at least 8");
    }
    this.fsDirectory = in;
    this.bufferBytes = bufferBytes;
    this.bufferedExtensions = Set.copyOf(bufferedExtensions);
    if (this.bufferedExtensions.isEmpty()) {
      throw new IllegalArgumentException("bufferedExtensions must not be empty");
    }
  }

  @Override
  public IndexOutput createOutput(String name, IOContext context) throws IOException {
    String extension = IndexFileNames.getExtension(name);
    if (extension == null || !bufferedExtensions.contains(extension)) {
      return in.createOutput(name, context);
    }
    OutputStream stream =
        Files.newOutputStream(
            fsDirectory.getDirectory().resolve(name),
            StandardOpenOption.WRITE,
            StandardOpenOption.CREATE_NEW);
    return new OutputStreamIndexOutput(
        "LargeVectorWriteDirectory(path=\"" + fsDirectory.getDirectory() + "\")",
        name,
        stream,
        bufferBytes);
  }

  private static Set<String> configuredExtensions() {
    String value = System.getProperty("benchmark.largeVectorWriteExtensions", "vec");
    Set<String> extensions =
        Arrays.stream(value.split(","))
            .map(String::trim)
            .filter(extension -> extension.isEmpty() == false)
            .collect(Collectors.toUnmodifiableSet());
    if (extensions.isEmpty()) {
      throw new IllegalArgumentException(
          "benchmark.largeVectorWriteExtensions must name at least one extension");
    }
    return extensions;
  }
}
