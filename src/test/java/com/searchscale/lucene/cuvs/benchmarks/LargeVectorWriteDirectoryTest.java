package com.searchscale.lucene.cuvs.benchmarks;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Set;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

public class LargeVectorWriteDirectoryTest {

  @Rule public final TemporaryFolder temporaryFolder = new TemporaryFolder();

  @Test
  public void buffersConfiguredVectorExtensionsAndDelegatesOtherFiles() throws Exception {
    Path path = temporaryFolder.newFolder("index").toPath();
    byte[] payload = new byte[3 * 1024 * 1024 + 17];
    for (int i = 0; i < payload.length; i++) {
      payload[i] = (byte) i;
    }

    try (LargeVectorWriteDirectory directory =
        new LargeVectorWriteDirectory(
            FSDirectory.open(path), 128 * 1024, Set.of("vec", "vex"))) {
      write(directory, "segment.vec", payload);
      write(directory, "segment.vex", payload);
      try (IndexOutput output = directory.createOutput("segment.vem", IOContext.DEFAULT)) {
        output.writeInt(7);
      }
      try (IndexOutput output = directory.createOutput("pending_segments_1", IOContext.DEFAULT)) {
        output.writeInt(11);
      }
      directory.sync(
          List.of("segment.vec", "segment.vex", "segment.vem", "pending_segments_1"));

      assertArrayEquals(payload, read(directory, "segment.vec"));
      assertArrayEquals(payload, read(directory, "segment.vex"));
      assertThrows(
          IOException.class, () -> directory.createOutput("segment.vec", IOContext.DEFAULT));
    }
  }

  private static void write(
      LargeVectorWriteDirectory directory, String fileName, byte[] payload) throws IOException {
    try (IndexOutput output = directory.createOutput(fileName, IOContext.DEFAULT)) {
      output.writeBytes(payload, 0, payload.length);
    }
  }

  private static byte[] read(LargeVectorWriteDirectory directory, String fileName)
      throws IOException {
    try (IndexInput input = directory.openInput(fileName, IOContext.DEFAULT)) {
      byte[] result = new byte[Math.toIntExact(input.length())];
      input.readBytes(result, 0, result.length);
      assertEquals(result.length, input.length());
      return result;
    }
  }
}
