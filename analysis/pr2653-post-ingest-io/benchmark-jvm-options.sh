#!/usr/bin/env bash

# Shared JVM sizing for launchers that run the benchmark either directly or inside Maven.
configure_benchmark_jvm() {
  BENCHMARK_JAVA_INITIAL_HEAP=${JAVA_INITIAL_HEAP:-64g}
  BENCHMARK_JAVA_MAX_HEAP=${JAVA_MAX_HEAP:-256g}

  local heap_pattern='^[1-9][0-9]*[kKmMgG]$'
  if ! [[ "$BENCHMARK_JAVA_INITIAL_HEAP" =~ $heap_pattern ]]; then
    echo "JAVA_INITIAL_HEAP must be a positive JVM size such as 16g; got $BENCHMARK_JAVA_INITIAL_HEAP" >&2
    return 2
  fi
  if ! [[ "$BENCHMARK_JAVA_MAX_HEAP" =~ $heap_pattern ]]; then
    echo "JAVA_MAX_HEAP must be a positive JVM size such as 96g; got $BENCHMARK_JAVA_MAX_HEAP" >&2
    return 2
  fi

  local initial_bytes max_bytes
  initial_bytes=$(benchmark_heap_size_bytes "$BENCHMARK_JAVA_INITIAL_HEAP")
  max_bytes=$(benchmark_heap_size_bytes "$BENCHMARK_JAVA_MAX_HEAP")
  if [ "$initial_bytes" -gt "$max_bytes" ]; then
    echo "JAVA_INITIAL_HEAP must not exceed JAVA_MAX_HEAP" >&2
    return 2
  fi

  BENCHMARK_JAVA_OPTIONS=(
    "-Xms$BENCHMARK_JAVA_INITIAL_HEAP"
    "-Xmx$BENCHMARK_JAVA_MAX_HEAP"
    -XX:+UseG1GC
    -XX:MinHeapFreeRatio=5
    -XX:MaxHeapFreeRatio=15
    -XX:G1PeriodicGCInterval=5000
    -XX:+ExitOnOutOfMemoryError
    --add-modules=jdk.incubator.vector
    --enable-native-access=ALL-UNNAMED
  )
}

benchmark_heap_size_bytes() {
  local size=$1
  local number=${size%?}
  case "${size: -1}" in
    k|K) printf '%s\n' "$((number * 1024))" ;;
    m|M) printf '%s\n' "$((number * 1024 * 1024))" ;;
    g|G) printf '%s\n' "$((number * 1024 * 1024 * 1024))" ;;
  esac
}

configure_benchmark_maven_jvm() {
  configure_benchmark_jvm || return
  MAVEN_OPTS="${MAVEN_OPTS:+$MAVEN_OPTS }${BENCHMARK_JAVA_OPTIONS[*]}"
  export MAVEN_OPTS
}
