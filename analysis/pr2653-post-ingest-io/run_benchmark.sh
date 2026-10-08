#!/usr/bin/env bash

set -euo pipefail

if [ "$#" -ne 4 ]; then
  echo "Usage: $0 <config-template.json> <results-dir> <benchmark-id> <cuvs-worktree>" >&2
  exit 2
fi

config_template=$(realpath "$1")
results_dir=$(realpath -m "$2")
benchmark_id=$3
cuvs_root=$(realpath "$4")
script_dir=$(cd -- "$(dirname -- "$0")" && pwd)
benchmark_root=$(git -C "$script_dir" rev-parse --show-toplevel)
actual_cuvs_head=$(git -C "$cuvs_root" rev-parse HEAD)
if [ -n "${CUVS_SELECTED_HEAD:-}" ] && [ "$actual_cuvs_head" != "$CUVS_SELECTED_HEAD" ]; then
  echo "cuVS HEAD changed before benchmark: expected $CUVS_SELECTED_HEAD, got $actual_cuvs_head" >&2
  exit 4
fi
source "$script_dir/benchmark-jvm-options.sh"
# This runner's largest allocation is the native host vector matrix. Keep enough Java heap for one
# bounded 60 GiB DWPT while preserving host memory for that matrix.
JAVA_INITIAL_HEAP=${JAVA_INITIAL_HEAP:-16g}
JAVA_MAX_HEAP=${JAVA_MAX_HEAP:-96g}
configure_benchmark_jvm
source_cache_mode=${SOURCE_CACHE_MODE:-warm}
cleanup_root=$(realpath -m "${INDEX_CLEANUP_ROOT:-${HOME:?HOME must be set}/index_build_dir/cuvs-benchmark}")
index_dir="$cleanup_root/$benchmark_id"
host_lock_file=${BENCHMARK_HOST_LOCK_FILE:-/mnt/local-nvme/benchmark-results/cuvs-benchmark.lock}
drop_file_cache_bin=${DROP_FILE_CACHE_BIN:-/tmp/wikiann_drop_file_cache}
proactive_vector_sync=${PROACTIVE_VECTOR_SYNC:-false}
proactive_sync_interval=${PROACTIVE_VECTOR_SYNC_INTERVAL_SEC:-0.1}
proactive_sync_start_cutoff_fraction=${PROACTIVE_VECTOR_SYNC_START_CUTOFF_FRACTION:-}
proactive_sync_helper_source=${PROACTIVE_VECTOR_SYNC_HELPER_SOURCE:-"$script_dir/proactive_range_writeback.sh"}
proactive_sync_helper="$results_dir/proactive-vector-sync-helper-used.sh"
proactive_sync_bin=${PROACTIVE_VECTOR_SYNC_BIN:-"$script_dir/sync_file_range_writeback"}
output_quiescence_max_kb=${OUTPUT_QUIESCENCE_MAX_KB:-262144}
output_quiescence_timeout_sec=${OUTPUT_QUIESCENCE_TIMEOUT_SEC:-300}
vector_row_copy_mode=${VECTOR_ROW_COPY_MODE:-bulk}
require_idle_gpu=${BENCHMARK_REQUIRE_IDLE_GPU:-true}
expect_hidden_gpu_control=${CUVS_EXPECT_HIDDEN_GPU_CONTROL:-false}
frozen_native_helper=${CUVS_FROZEN_NATIVE_HELPER:-}
native_artifact_dir=${CUVS_NATIVE_ARTIFACT_DIR:-}
native_dependency_manifest=${CUVS_NATIVE_DEPENDENCY_MANIFEST:-}
prewarm_index_before_search=${PREWARM_INDEX_BEFORE_SEARCH:-}
enable_index_writer_info_stream=${ENABLE_INDEX_WRITER_INFO_STREAM:-}
post_ingest_overlap=${CUVS_POST_INGEST_OVERLAP:-false}
large_vector_write_extensions=${LARGE_VECTOR_WRITE_EXTENSIONS:-vec}
required_index_free_bytes=${REQUIRED_INDEX_FREE_BYTES:-}
required_index_free_bytes_source=environment

if ! command -v flock >/dev/null; then
  echo "Missing required host-lock executable: flock" >&2
  exit 4
fi
mkdir -p "$(dirname "$host_lock_file")"
exec 9>"$host_lock_file"
if ! flock -n 9; then
  echo "Another benchmark holds the host lock: $host_lock_file" >&2
  exit 8
fi

case "$source_cache_mode" in
  cold|warm|observe) ;;
  *)
    echo "SOURCE_CACHE_MODE must be cold, warm, or observe; got $source_cache_mode" >&2
    exit 2
    ;;
esac
case "$proactive_vector_sync" in
  true|false) ;;
  *)
    echo "PROACTIVE_VECTOR_SYNC must be true or false; got $proactive_vector_sync" >&2
    exit 2
    ;;
esac
if [ -n "$proactive_sync_start_cutoff_fraction" ] && [ "$proactive_vector_sync" != true ]; then
  echo "PROACTIVE_VECTOR_SYNC_START_CUTOFF_FRACTION requires PROACTIVE_VECTOR_SYNC=true" >&2
  exit 2
fi
case "$vector_row_copy_mode" in
  bulk|scalar) ;;
  *)
    echo "VECTOR_ROW_COPY_MODE must be bulk or scalar; got $vector_row_copy_mode" >&2
    exit 2
    ;;
esac
case "$require_idle_gpu" in
  true|false) ;;
  *)
    echo "BENCHMARK_REQUIRE_IDLE_GPU must be true or false; got $require_idle_gpu" >&2
    exit 2
    ;;
esac
case "$expect_hidden_gpu_control" in
  true|false) ;;
  *)
    echo "CUVS_EXPECT_HIDDEN_GPU_CONTROL must be true or false; got $expect_hidden_gpu_control" >&2
    exit 2
    ;;
esac
case "$post_ingest_overlap" in
  true|false) ;;
  *)
    echo "CUVS_POST_INGEST_OVERLAP must be true or false; got $post_ingest_overlap" >&2
    exit 2
    ;;
esac
if ! [[ "$large_vector_write_extensions" =~ ^(vec|vex)(,(vec|vex))*$ ]]; then
  echo "LARGE_VECTOR_WRITE_EXTENSIONS must be a comma-separated subset of vec,vex" >&2
  exit 2
fi
if [ -n "${PARTITIONED_OVERLAP:-}" ]; then
  case "$PARTITIONED_OVERLAP" in
    true|false) ;;
    *)
      echo "PARTITIONED_OVERLAP must be true or false; got $PARTITIONED_OVERLAP" >&2
      exit 2
      ;;
  esac
fi
if [ -n "${PARTITIONED_SERIAL_INGEST_PIPELINE:-}" ]; then
  case "$PARTITIONED_SERIAL_INGEST_PIPELINE" in
    true|false) ;;
    *)
      echo "PARTITIONED_SERIAL_INGEST_PIPELINE must be true or false; got $PARTITIONED_SERIAL_INGEST_PIPELINE" >&2
      exit 2
      ;;
  esac
fi
if [ -n "${DEFER_PARTITION_SYNC_UNTIL_COMBINE:-}" ]; then
  case "$DEFER_PARTITION_SYNC_UNTIL_COMBINE" in
    true|false) ;;
    *)
      echo "DEFER_PARTITION_SYNC_UNTIL_COMBINE must be true or false; got $DEFER_PARTITION_SYNC_UNTIL_COMBINE" >&2
      exit 2
      ;;
  esac
fi
if [ -n "${PARTITIONED_SLICE_WEIGHTS:-}" ] \
  && ! [[ "$PARTITIONED_SLICE_WEIGHTS" =~ ^[1-9][0-9]*(,[1-9][0-9]*)*$ ]]; then
  echo "PARTITIONED_SLICE_WEIGHTS must be comma-separated positive integers; got $PARTITIONED_SLICE_WEIGHTS" >&2
  exit 2
fi
if [ -n "$prewarm_index_before_search" ]; then
  case "$prewarm_index_before_search" in
    true|false) ;;
    *)
      echo "PREWARM_INDEX_BEFORE_SEARCH must be true or false; got $prewarm_index_before_search" >&2
      exit 2
      ;;
  esac
fi
if [ -n "$enable_index_writer_info_stream" ]; then
  case "$enable_index_writer_info_stream" in
    true|false) ;;
    *)
      echo "ENABLE_INDEX_WRITER_INFO_STREAM must be true or false; got $enable_index_writer_info_stream" >&2
      exit 2
      ;;
  esac
fi
if [ -n "${RECLAIM_HEAP_BEFORE_FORCE_MERGE:-}" ]; then
  case "$RECLAIM_HEAP_BEFORE_FORCE_MERGE" in
    true|false) ;;
    *)
      echo "RECLAIM_HEAP_BEFORE_FORCE_MERGE must be true or false; got $RECLAIM_HEAP_BEFORE_FORCE_MERGE" >&2
      exit 2
      ;;
  esac
fi
for positive_integer_override in \
  NUM_INDEX_THREADS \
  PARTITIONED_PIPELINE_DEPTH \
  CUVS_HOST_MATRIX_COPY_THREADS \
  INDEX_WRITER_PER_THREAD_HARD_LIMIT_MB \
  MAX_MERGED_SEGMENT_MB; do
  value=${!positive_integer_override:-}
  if [ -n "$value" ] && { ! [[ "$value" =~ ^[0-9]+$ ]] || [ "$value" -lt 1 ]; }; then
    echo "$positive_integer_override must be a positive integer; got $value" >&2
    exit 2
  fi
done
if ! [[ "${EXPECTED_SEGMENT_COUNT:-0}" =~ ^[0-9]+$ ]]; then
  echo "EXPECTED_SEGMENT_COUNT must be a non-negative integer; got ${EXPECTED_SEGMENT_COUNT:-}" >&2
  exit 2
fi
for loader_variable in \
  LD_PRELOAD LD_LIBRARY_PATH LD_AUDIT LD_DEBUG LD_DEBUG_OUTPUT LD_PROFILE \
  LD_PROFILE_OUTPUT LD_TRACE_LOADED_OBJECTS; do
  if [[ -v "$loader_variable" ]]; then
    echo "Unexpected inherited loader variable: $loader_variable" >&2
    exit 4
  fi
done
if [ -n "$required_index_free_bytes" ] && ! [[ "$required_index_free_bytes" =~ ^[0-9]+$ ]]; then
  echo "REQUIRED_INDEX_FREE_BYTES must be a non-negative integer; got $required_index_free_bytes" >&2
  exit 2
fi
if [ -n "${CUVS_HOST_MATRIX_COPY_THREADS:-}" ] && [ "$CUVS_HOST_MATRIX_COPY_THREADS" -gt 64 ]; then
  echo "CUVS_HOST_MATRIX_COPY_THREADS must be at most 64; got $CUVS_HOST_MATRIX_COPY_THREADS" >&2
  exit 2
fi

if [ -e "$results_dir" ] && [ -n "$(find "$results_dir" -mindepth 1 -print -quit)" ]; then
  echo "Refusing to overwrite non-empty results directory: $results_dir" >&2
  exit 3
fi
if [ -e "$index_dir" ] && [ -n "$(find "$index_dir" -mindepth 1 -print -quit)" ]; then
  echo "Refusing to benchmark into non-empty index directory: $index_dir" >&2
  exit 3
fi

if [ -z "$required_index_free_bytes" ]; then
  required_index_free_bytes_source=estimated
  estimate_num_docs=$(jq -er '.numDocs' "$config_template")
  estimate_dimensions=$(jq -er '.vectorDimension' "$config_template")
  estimate_graph_degree=$(jq -er '.cagraGraphDegree // .hnswMaxConn // 0' "$config_template")
  estimate_force_merge=$(jq -er '.forceMerge // 0' "$config_template")
  estimate_tiered_merge=$(jq -r '.enableTieredMerge // false' "$config_template")
  estimate_compound_file=$(
    jq -r 'if has("useCompoundFile") then .useCompoundFile else true end' "$config_template"
  )
  for estimate_value in \
    "$estimate_num_docs" \
    "$estimate_dimensions" \
    "$estimate_graph_degree" \
    "$estimate_force_merge"; do
    if ! [[ "$estimate_value" =~ ^[0-9]+$ ]]; then
      echo "Cannot estimate index space from non-integer configuration value: $estimate_value" >&2
      exit 2
    fi
  done
  estimated_index_bytes=$((
    estimate_num_docs * estimate_dimensions * 4
      + estimate_num_docs * estimate_graph_degree * 4
  ))
  estimated_concurrent_copies=1
  if [ "$estimate_force_merge" -gt 0 ] || [ "$estimate_tiered_merge" = true ]; then
    estimated_concurrent_copies=2
  fi
  if [ "$estimate_compound_file" = true ]; then
    estimated_concurrent_copies=$((estimated_concurrent_copies + 1))
  fi
  # Allow 10% for postings/metadata and 4 GiB for filesystem and merge bookkeeping.
  required_index_free_bytes=$((
    estimated_index_bytes * estimated_concurrent_copies * 11 / 10 + 4 * 1024 * 1024 * 1024
  ))
fi

mkdir -p "$results_dir" "$index_dir"
index_available_bytes=$(df --output=avail -B1 "$index_dir" | tail -1 | tr -d '[:space:]')
if [ "$index_available_bytes" -lt "$required_index_free_bytes" ]; then
  echo "Index filesystem has $index_available_bytes free bytes; $required_index_free_bytes required" >&2
  exit 5
fi
jq --arg index_dir "$index_dir" \
  --arg large_vector_buffer_kb "${LARGE_VECTOR_WRITE_BUFFER_KB:-}" \
  --arg measured_query_count "${MEASURED_QUERY_COUNT:-}" \
  --arg cuvs_writer_threads "${CUVS_WRITER_THREADS:-}" \
  --arg cuvs_host_matrix_copy_threads "${CUVS_HOST_MATRIX_COPY_THREADS:-}" \
  --arg num_index_threads "${NUM_INDEX_THREADS:-}" \
  --arg partitioned_overlap "${PARTITIONED_OVERLAP:-}" \
  --arg partitioned_serial_ingest_pipeline "${PARTITIONED_SERIAL_INGEST_PIPELINE:-}" \
  --arg partitioned_pipeline_depth "${PARTITIONED_PIPELINE_DEPTH:-}" \
  --arg defer_partition_sync_until_combine "${DEFER_PARTITION_SYNC_UNTIL_COMBINE:-}" \
  --arg partitioned_slice_weights "${PARTITIONED_SLICE_WEIGHTS:-}" \
  --arg prewarm_index_before_search "$prewarm_index_before_search" \
  --arg enable_index_writer_info_stream "$enable_index_writer_info_stream" \
  --arg index_writer_per_thread_hard_limit_mb "${INDEX_WRITER_PER_THREAD_HARD_LIMIT_MB:-}" \
  --arg max_merged_segment_mb "${MAX_MERGED_SEGMENT_MB:-}" \
  --arg reclaim_heap_before_force_merge "${RECLAIM_HEAP_BEFORE_FORCE_MERGE:-}" \
  --arg expected_segment_count "${EXPECTED_SEGMENT_COUNT:-}" \
  --arg ivf_pq_dim "${IVF_PQ_DIM:-}" \
  --arg ivf_pq_n_lists "${IVF_PQ_N_LISTS:-}" \
  --arg ivf_pq_n_probes "${IVF_PQ_N_PROBES:-}" \
  --arg ivf_pq_train_fraction "${IVF_PQ_TRAINSET_FRACTION:-}" \
  '.indexDirPath = $index_dir
   | if $large_vector_buffer_kb == "" then .
     else .useLargeVectorWriteOutput = true
       | .largeVectorWriteBufferSizeKB = ($large_vector_buffer_kb | tonumber)
     end
   | if $measured_query_count == "" then .
     else .numQueriesToRun = (.numWarmUpQueries + ($measured_query_count | tonumber))
     end
   | if $cuvs_writer_threads == "" then .
     else .cuvsWriterThreads = ($cuvs_writer_threads | tonumber)
     end
   | if $cuvs_host_matrix_copy_threads == "" then .
     else .cuvsHostMatrixCopyThreads = ($cuvs_host_matrix_copy_threads | tonumber)
     end
   | if $num_index_threads == "" then .
     else .numIndexThreads = ($num_index_threads | tonumber)
     end
   | if $partitioned_overlap == "" then .
     else .partitionedOverlap = ($partitioned_overlap == "true")
     end
   | if $partitioned_serial_ingest_pipeline == "" then .
     else .partitionedSerialIngestPipeline = ($partitioned_serial_ingest_pipeline == "true")
     end
   | if $partitioned_pipeline_depth == "" then .
     else .partitionedPipelineDepth = ($partitioned_pipeline_depth | tonumber)
     end
   | if $defer_partition_sync_until_combine == "" then .
     else .deferPartitionSyncUntilCombine = ($defer_partition_sync_until_combine == "true")
     end
   | if $partitioned_slice_weights == "" then .
     else .partitionedSliceWeights = ($partitioned_slice_weights | split(",") | map(tonumber))
     end
   | if $prewarm_index_before_search == "" then .
     else .prewarmIndexBeforeSearch = ($prewarm_index_before_search == "true")
     end
   | if $enable_index_writer_info_stream == "" then .
     else .enableIndexWriterInfoStream = ($enable_index_writer_info_stream == "true")
     end
   | if $index_writer_per_thread_hard_limit_mb == "" then .
     else .indexWriterPerThreadHardLimitMB = ($index_writer_per_thread_hard_limit_mb | tonumber)
     end
   | if $max_merged_segment_mb == "" then .
     else .maxMergedSegmentMB = ($max_merged_segment_mb | tonumber)
     end
   | if $reclaim_heap_before_force_merge == "" then .
     else .reclaimHeapBeforeForceMerge = ($reclaim_heap_before_force_merge == "true")
     end
   | if $expected_segment_count == "" then .
     else .expectedSegmentCount = ($expected_segment_count | tonumber)
     end
   | if $ivf_pq_dim == "" then .
     else .cuVSIvfPqIndexParamsPqDim = ($ivf_pq_dim | tonumber)
     end
   | if $ivf_pq_n_lists == "" then .
     else .cuVSIvfPqIndexParamsNLists = ($ivf_pq_n_lists | tonumber)
     end
   | if $ivf_pq_n_probes == "" then .
     else .cuVSIvfPqSearchParamsNProbes = ($ivf_pq_n_probes | tonumber)
     end
   | if $ivf_pq_train_fraction == "" then .
     else .cuVSIvfPqIndexParamsKmeansTrainsetFraction = ($ivf_pq_train_fraction | tonumber)
     end' \
  "$config_template" > "$results_dir/config.json"
cp "$0" "$results_dir/runner-used.sh"
cp "$script_dir/benchmark-jvm-options.sh" "$results_dir/benchmark-jvm-options-used.sh"
mkdir "$results_dir/benchmark-source-used"
cp "$benchmark_root/pom.xml" "$results_dir/benchmark-source-used/pom.xml"
cp -a "$benchmark_root/src/main" "$results_dir/benchmark-source-used/main"

config_file="$results_dir/config.json"
dataset_file=$(jq -er '.datasetFile' "$config_file")
query_file=$(jq -er '.queryFile' "$config_file")
ground_truth_file=$(jq -er '.groundTruthFile' "$config_file")
proactive_ready_file="$results_dir/proactive-vector-sync-ingest-complete.tsv"
proactive_manifest_file="$results_dir/proactive-vector-sync-segments.tsv"
proactive_stop_file="$results_dir/.stop-proactive-vector-sync"
proactive_events_file="$results_dir/proactive-vector-sync.tsv"
proactive_sync_expected_payload_bytes=""
proactive_sync_start_cutoff_bytes=""
proactive_sync_helper_mode=disabled
if [ "$proactive_vector_sync" = true ]; then
  if [ ! -x "$proactive_sync_helper_source" ]; then
    echo "Missing executable proactive-sync helper: $proactive_sync_helper_source" >&2
    exit 4
  fi
  cp -- "$proactive_sync_helper_source" "$proactive_sync_helper"
  chmod 0555 "$proactive_sync_helper"
  if [ ! -x "$proactive_sync_helper" ]; then
    echo "Proactive-sync helper snapshot is not executable: $proactive_sync_helper" >&2
    exit 4
  fi
  if ! proactive_sync_bin=$(command -v "$proactive_sync_bin"); then
    echo "Missing proactive-sync executable: ${PROACTIVE_VECTOR_SYNC_BIN:-sync}" >&2
    exit 4
  fi
  if ! awk -v interval="$proactive_sync_interval" \
    'BEGIN { exit !(interval > 0 && interval <= 5) }'; then
    echo "PROACTIVE_VECTOR_SYNC_INTERVAL_SEC must be greater than 0 and at most 5" >&2
    exit 2
  fi
  if [ -n "$proactive_sync_start_cutoff_fraction" ]; then
    "$proactive_sync_helper" validate "$config_file" "$proactive_sync_start_cutoff_fraction"
    proactive_num_docs=$(jq -er '.numDocs' "$config_file")
    proactive_vector_dimension=$(jq -er '.vectorDimension' "$config_file")
    proactive_sync_expected_payload_bytes=$((proactive_num_docs * proactive_vector_dimension * 4))
    proactive_sync_start_cutoff_bytes=$(
      awk -v expected="$proactive_sync_expected_payload_bytes" \
        -v fraction="$proactive_sync_start_cutoff_fraction" \
        'BEGIN { value = int(expected * fraction); print (value > 0 ? value : 1) }'
    )
  else
    "$proactive_sync_helper" validate "$config_file"
  fi
  if [ "$(jq -r '(.partitionedOverlap // false) or (.partitionedSerialIngestPipeline // false)' "$config_file")" = true ]; then
    proactive_sync_helper_mode=partitioned-overlap-manifest-v1
  else
    proactive_sync_helper_mode=index-directory-v1
  fi
fi
cuvs_lucene_jar=${CUVS_LUCENE_JAR:-$cuvs_root/java/cuvs-lucene/target/cuvs-lucene-26.12.0.jar}
cuvs_java_jar=${CUVS_JAVA_JAR:-$cuvs_root/java/cuvs-java/target/cuvs-java-26.12.0.jar}
cuvs_java_jar=$(realpath "$cuvs_java_jar")
native_dir=$(realpath "${CUVS_NATIVE_DIR:-/home/ec2-user/miniforge3/envs/cuvs-dev/lib}")
libcuvs=$(realpath "$native_dir/libcuvs.so")
libcuvs_c=$(realpath "$native_dir/libcuvs_c.so")
if [ -z "$frozen_native_helper" ] || [ -z "$native_artifact_dir" ] \
    || [ -z "$native_dependency_manifest" ]; then
  echo "Frozen-native helper, artifact directory, and dependency manifest are required" >&2
  exit 4
fi
frozen_native_helper=$(realpath "$frozen_native_helper")
native_artifact_dir=$(realpath "$native_artifact_dir")
native_dependency_manifest=$(realpath "$native_dependency_manifest")

assert_artifact_fresh() {
  local artifact=$1
  local label=$2
  shift 2
  local source newer_source
  for source in "$@"; do
    if [ -d "$source" ]; then
      newer_source=$(find "$source" -type f -newer "$artifact" -print -quit)
      if [ -n "$newer_source" ]; then
        echo "$label is stale: $newer_source is newer than $artifact" >&2
        exit 4
      fi
    elif [ -f "$source" ] && [ "$source" -nt "$artifact" ]; then
      echo "$label is stale: $source is newer than $artifact" >&2
      exit 4
    fi
  done
}

for required in \
  "$dataset_file" \
  "$query_file" \
  "$ground_truth_file" \
  "$cuvs_lucene_jar" \
  "$cuvs_java_jar" \
  "$libcuvs" \
  "$libcuvs_c" \
  "$frozen_native_helper" \
  "$native_dependency_manifest" \
  "$benchmark_root/target/classes/com/searchscale/lucene/cuvs/benchmarks/LuceneCuvsBenchmarks.class"; do
  if [ ! -f "$required" ]; then
    echo "Required artifact is missing: $required" >&2
    exit 4
  fi
done

assert_artifact_fresh \
  "$cuvs_lucene_jar" \
  "cuVS-Lucene JAR" \
  "$cuvs_root/java/cuvs-lucene/src" \
  "$cuvs_root/java/cuvs-lucene/pom.xml"
assert_artifact_fresh \
  "$cuvs_java_jar" \
  "cuVS-Java JAR" \
  "$cuvs_root/java/cuvs-java/src" \
  "$cuvs_root/java/cuvs-java/pom.xml"
assert_artifact_fresh \
  "$benchmark_root/target/classes/com/searchscale/lucene/cuvs/benchmarks/LuceneCuvsBenchmarks.class" \
  "benchmark classes" \
  "$benchmark_root/src/main/java" \
  "$benchmark_root/pom.xml"

raw_classpath="$results_dir/maven-runtime-classpath.txt"
maven_version_args=()
if [ -n "${CUVS_MAVEN_VERSION:-}" ]; then
  if ! [[ "$CUVS_MAVEN_VERSION" =~ ^[0-9A-Za-z._-]+$ ]]; then
    echo "CUVS_MAVEN_VERSION contains unsupported characters: $CUVS_MAVEN_VERSION" >&2
    exit 2
  fi
  maven_version_args+=("-Dcuvs.version=$CUVS_MAVEN_VERSION")
fi
mvn -q -f "$benchmark_root/pom.xml" "${maven_version_args[@]}" \
  -DincludeScope=runtime dependency:build-classpath -Dmdep.outputFile="$raw_classpath"
runtime_classpath="$benchmark_root/target/classes:$cuvs_lucene_jar:$cuvs_java_jar"
while IFS= read -r entry || [ -n "$entry" ]; do
  case "$(basename "$entry")" in
    cuvs-lucene-*.jar|cuvs-java-*.jar) continue ;;
  esac
  runtime_classpath="$runtime_classpath:$entry"
done < <(tr ':' '\n' < "$raw_classpath" | sed '/^$/d')
printf '%s\n' "$runtime_classpath" > "$results_dir/effective-classpath.txt"
(
  cd "$benchmark_root/target/classes"
  find . -type f -print0 | sort -z | xargs -0 sha256sum
) > "$results_dir/benchmark-classes-sha256.txt"
benchmark_classes_sha256=$(sha256sum "$results_dir/benchmark-classes-sha256.txt" | awk '{print $1}')

if ! [[ "$output_quiescence_max_kb" =~ ^[0-9]+$ ]] \
  || ! [[ "$output_quiescence_timeout_sec" =~ ^[0-9]+$ ]]; then
  echo "Output quiescence limits must be non-negative integers" >&2
  exit 2
fi

quiescence_start_epoch=$(date +%s)
printf 'timestamp_s\tdirty_kb\twriteback_kb\ttotal_kb\n' \
  > "$results_dir/output-quiescence.tsv"
while true; do
  read -r dirty_kb writeback_kb < <(
    awk '/^Dirty:/ {d=$2} /^Writeback:/ {w=$2} END {print d + 0, w + 0}' /proc/meminfo
  )
  total_kb=$((dirty_kb + writeback_kb))
  printf '%s\t%s\t%s\t%s\n' "$(date +%s.%N)" "$dirty_kb" "$writeback_kb" "$total_kb" \
    >> "$results_dir/output-quiescence.tsv"
  if [ "$total_kb" -le "$output_quiescence_max_kb" ]; then
    break
  fi
  if [ "$(( $(date +%s) - quiescence_start_epoch ))" -ge "$output_quiescence_timeout_sec" ]; then
    echo "Output did not quiesce below ${output_quiescence_max_kb} KiB in ${output_quiescence_timeout_sec}s" >&2
    exit 5
  fi
  sleep 1
done

record_cache_state() {
  local output=$1
  fincore -b "$dataset_file" > "$output"
}

resident_bytes() {
  fincore -b -n -o RES "$dataset_file" | tr -d '[:space:]'
}

record_cache_state "$results_dir/source-cache-unconditioned.txt"
cache_start_ns=$(date +%s%N)
case "$source_cache_mode" in
  cold)
    if [ ! -x "$drop_file_cache_bin" ]; then
      echo "Missing file-cache eviction helper: $drop_file_cache_bin" >&2
      exit 5
    fi
    sync -d "$dataset_file"
    "$drop_file_cache_bin" "$dataset_file"
    ;;
  warm)
    dd if="$dataset_file" of=/dev/null bs=16M iflag=fullblock status=none
    ;;
  observe) ;;
esac
cache_end_ns=$(date +%s%N)
record_cache_state "$results_dir/source-cache-before.txt"
resident=$(resident_bytes)
source_size=$(stat -c '%s' "$dataset_file")
printf 'mode\tduration_seconds\tresident_bytes\tsize_bytes\tresident_fraction\n' \
  > "$results_dir/source-cache-conditioning.tsv"
awk -v mode="$source_cache_mode" -v elapsed="$((cache_end_ns - cache_start_ns))" \
  -v resident="$resident" -v size="$source_size" \
  'BEGIN { printf "%s\t%.9f\t%d\t%d\t%.9f\n", mode, elapsed / 1e9, resident, size, resident / size }' \
  >> "$results_dir/source-cache-conditioning.tsv"
if [ "$source_cache_mode" = cold ] && [ "$resident" -ne 0 ]; then
  echo "Cold conditioning left $resident of $source_size source bytes resident" >&2
  exit 5
fi
if [ "$source_cache_mode" = warm ] && [ "$resident" -lt "$source_size" ]; then
  echo "Warm conditioning left only $resident of $source_size source bytes resident" >&2
  exit 5
fi

gpu_compute_snapshot="$results_dir/gpu-compute-before.csv"
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory \
  --format=csv,noheader,nounits > "$gpu_compute_snapshot"
if [ "$require_idle_gpu" = true ] && [ -s "$gpu_compute_snapshot" ]; then
  echo "GPU is not idle after source-cache conditioning:" >&2
  cat "$gpu_compute_snapshot" >&2
  exit 9
fi

df -B1 "$dataset_file" "$index_dir" > "$results_dir/filesystems-before.txt"
stat -f "$dataset_file" "$index_dir" > "$results_dir/filesystem-stats-before.txt"

{
  echo "benchmark_repo=$(git -C "$benchmark_root" rev-parse HEAD)"
  echo "cuvs_repo=$(git -C "$cuvs_root" rev-parse HEAD)"
  echo "cuvs_selected_head=${CUVS_SELECTED_HEAD:-unspecified}"
  echo "cuvs_native_mode=${CUVS_NATIVE_MODE:-unspecified}"
  echo "cuvs_build_root=${CUVS_BUILD_ROOT:-unspecified}"
  echo "cuvs_maven_version=${CUVS_MAVEN_VERSION:-unspecified}"
  echo "benchmark_wrapper=${CUVS_BENCHMARK_WRAPPER:-unspecified}"
  if [ -n "${CUVS_BENCHMARK_WRAPPER:-}" ] && [ -f "$CUVS_BENCHMARK_WRAPPER" ]; then
    echo "benchmark_wrapper_sha256=$(sha256sum "$CUVS_BENCHMARK_WRAPPER" | awk '{print $1}')"
    cp "$CUVS_BENCHMARK_WRAPPER" "$results_dir/wrapper-used.sh"
  fi
  echo "started_at=$(date --iso-8601=seconds)"
  echo "source_cache_mode=$source_cache_mode"
  echo "host_lock_file=$host_lock_file"
  echo "require_idle_gpu=$require_idle_gpu"
  echo "expect_hidden_gpu_control=$expect_hidden_gpu_control"
  echo "vector_row_copy_mode=$vector_row_copy_mode"
  echo "post_ingest_overlap=$post_ingest_overlap"
  echo "large_vector_write_extensions=$large_vector_write_extensions"
  echo "java_initial_heap=$BENCHMARK_JAVA_INITIAL_HEAP"
  echo "java_max_heap=$BENCHMARK_JAVA_MAX_HEAP"
  echo "java_options=${BENCHMARK_JAVA_OPTIONS[*]}"
  echo "required_index_free_bytes=$required_index_free_bytes"
  echo "required_index_free_bytes_source=$required_index_free_bytes_source"
  echo "index_available_bytes_before_run=$index_available_bytes"
  echo "benchmark_classes_manifest_sha256=$benchmark_classes_sha256"
  echo "output_quiescence_max_kb=$output_quiescence_max_kb"
  echo "output_quiescence_timeout_seconds=$output_quiescence_timeout_sec"
  echo "proactive_vector_sync=$proactive_vector_sync"
  echo "proactive_vector_sync_helper_mode=$proactive_sync_helper_mode"
  echo "proactive_vector_sync_interval_seconds=$proactive_sync_interval"
  echo "proactive_vector_sync_start_cutoff_fraction=${proactive_sync_start_cutoff_fraction:-disabled}"
  if [ "$proactive_vector_sync" = true ]; then
    echo "proactive_vector_sync_helper_source=$proactive_sync_helper_source"
    echo "proactive_vector_sync_helper=$proactive_sync_helper"
    echo "proactive_vector_sync_helper_sha256=$(sha256sum "$proactive_sync_helper" | awk '{print $1}')"
    echo "proactive_vector_sync_bin=$proactive_sync_bin"
    echo "proactive_vector_sync_bin_sha256=$(sha256sum "$proactive_sync_bin" | awk '{print $1}')"
    echo "proactive_range_writeback_chunk_bytes=${PROACTIVE_RANGE_WRITEBACK_CHUNK_BYTES:-disabled}"
    echo "proactive_range_writeback_guard_bytes=${PROACTIVE_RANGE_WRITEBACK_GUARD_BYTES:-disabled}"
    if [ "$proactive_sync_helper_mode" = partitioned-overlap-manifest-v1 ]; then
      echo "proactive_vector_sync_manifest=$proactive_manifest_file"
    fi
    if [ -n "$proactive_sync_start_cutoff_fraction" ]; then
      echo "proactive_vector_sync_expected_float32_payload_bytes=$proactive_sync_expected_payload_bytes"
      echo "proactive_vector_sync_start_cutoff_bytes=$proactive_sync_start_cutoff_bytes"
    fi
  fi
  echo "config_sha256=$(sha256sum "$config_file" | awk '{print $1}')"
  echo "dataset_identity=$(stat -c '%d:%i:%s:%Y' "$dataset_file")"
  echo "query_identity=$(stat -c '%d:%i:%s:%Y' "$query_file")"
  echo "query_sha256=$(sha256sum "$query_file" | awk '{print $1}')"
  echo "ground_truth_identity=$(stat -c '%d:%i:%s:%Y' "$ground_truth_file")"
  echo "ground_truth_sha256=$(sha256sum "$ground_truth_file" | awk '{print $1}')"
  if [ -x "$drop_file_cache_bin" ]; then
    echo "drop_file_cache_bin=$drop_file_cache_bin"
    echo "drop_file_cache_sha256=$(sha256sum "$drop_file_cache_bin" | awk '{print $1}')"
  fi
  echo "cuvs_lucene_jar=$cuvs_lucene_jar"
  echo "cuvs_lucene_jar_sha256=$(sha256sum "$cuvs_lucene_jar" | awk '{print $1}')"
  echo "cuvs_java_jar=$cuvs_java_jar"
  echo "cuvs_java_jar_sha256=$(sha256sum "$cuvs_java_jar" | awk '{print $1}')"
  echo "libcuvs=$libcuvs"
  echo "libcuvs_sha256=$(sha256sum "$libcuvs" | awk '{print $1}')"
  echo "libcuvs_c=$libcuvs_c"
  echo "libcuvs_c_sha256=$(sha256sum "$libcuvs_c" | awk '{print $1}')"
  echo "frozen_native_helper=$frozen_native_helper"
  echo "frozen_native_helper_sha256=$(sha256sum "$frozen_native_helper" | awk '{print $1}')"
  echo "native_dependency_manifest=$native_dependency_manifest"
  echo "native_dependency_manifest_sha256=$(sha256sum "$native_dependency_manifest" | awk '{print $1}')"
  if [ -e /etc/ld.so.preload ] || [ -L /etc/ld.so.preload ]; then
    echo "system_loader_preload=present"
  else
    echo "system_loader_preload=absent"
  fi
  findmnt -T "$dataset_file" -o SOURCE,TARGET,FSTYPE,OPTIONS -n
  findmnt -T "$index_dir" -o SOURCE,TARGET,FSTYPE,OPTIONS -n
  java -version 2>&1 | head -n 1
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
} > "$results_dir/provenance.txt"
git -C "$benchmark_root" status --short --untracked-files=all > "$results_dir/benchmark-git-status.txt"
git -C "$benchmark_root" diff --binary HEAD > "$results_dir/benchmark-working-tree.patch"
git -C "$cuvs_root" status --short --untracked-files=all > "$results_dir/cuvs-git-status.txt"
git -C "$cuvs_root" diff --binary HEAD > "$results_dir/cuvs-working-tree.patch"

monitor_pids=()
nvidia-smi \
  --query-gpu=timestamp,memory.used,utilization.gpu,utilization.memory,power.draw,temperature.gpu,clocks.current.sm,clocks.current.memory,pstate,clocks_throttle_reasons.active \
  --format=csv,noheader,nounits --loop-ms=200 > "$results_dir/gpu-samples.csv" &
monitor_pids+=("$!")
(
  printf 'timestamp_s\tpid\tprocess_name\tused_gpu_memory_mib\n'
  while true; do
    timestamp=$(date +%s.%N)
    while IFS= read -r process; do
      [ -n "$process" ] || continue
      printf '%s\t%s\n' "$timestamp" "$(printf '%s' "$process" | sed 's/, /\t/g')"
    done < <(
      nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory \
        --format=csv,noheader,nounits
    )
    sleep 0.5
  done
) > "$results_dir/gpu-compute-processes.tsv" &
monitor_pids+=("$!")
if command -v iostat >/dev/null; then
  iostat -y -tdxm 1 > "$results_dir/iostat-dxm.txt" &
  monitor_pids+=("$!")
fi
vmstat -w 1 > "$results_dir/vmstat.txt" &
monitor_pids+=("$!")
(
  printf 'timestamp_s\tDirty_kB\tWriteback_kB\tCached_kB\tMemAvailable_kB\n'
  while true; do
    printf '%s\t' "$(date +%s.%N)"
    awk '/^Dirty:/ {d=$2} /^Writeback:/ {w=$2} /^Cached:/ {c=$2} /^MemAvailable:/ {a=$2} END {print d "\t" w "\t" c "\t" a}' /proc/meminfo
    sleep 0.5
  done
) > "$results_dir/memory-writeback.tsv" &
monitor_pids+=("$!")
(
  printf 'timestamp_s\tvec_bytes\tvex_bytes\tcfs_bytes\n'
  while true; do
    printf '%s' "$(date +%s.%N)"
    for extension in vec vex cfs; do
      bytes=$(find "$index_dir" -maxdepth 1 -type f -name "*.$extension" -printf '%s\n' 2>/dev/null \
        | awk '{sum += $1} END {print sum + 0}')
      printf '\t%s' "$bytes"
    done
    printf '\n'
    sleep 1
  done
) > "$results_dir/index-growth.tsv" &
monitor_pids+=("$!")

stop_monitors() {
  for pid in "${monitor_pids[@]}"; do kill "$pid" 2>/dev/null || true; done
  for pid in "${monitor_pids[@]}"; do wait "$pid" 2>/dev/null || true; done
  monitor_pids=()
}

proactive_sync_pid=""
proactive_sync_spawn_ns=""
proactive_sync_stop_ns=""
proactive_sync_reaped_ns=""
proactive_sync_status=""
java_start_ns=""
java_end_ns=""

summarize_proactive_sync() {
  [ -f "$proactive_events_file" ] || return 0
  "$proactive_sync_helper" summarize \
    "$proactive_events_file" "$results_dir/proactive-vector-sync-summary.tsv"
}

write_proactive_lifecycle() {
  [ "$proactive_vector_sync" = true ] || return 0
  printf 'helper_spawn_ns\tjava_start_ns\tjava_end_ns\tstop_requested_ns\thelper_reaped_ns\thelper_status\n' \
    > "$results_dir/proactive-vector-sync-lifecycle.tsv"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$proactive_sync_spawn_ns" "$java_start_ns" "$java_end_ns" "$proactive_sync_stop_ns" \
    "$proactive_sync_reaped_ns" "$proactive_sync_status" \
    >> "$results_dir/proactive-vector-sync-lifecycle.tsv"
}

stop_proactive_sync() {
  [ -n "$proactive_sync_pid" ] || return 0
  proactive_sync_stop_ns=$(date +%s%N)
  touch "$proactive_stop_file"
  if wait "$proactive_sync_pid"; then
    proactive_sync_status=0
  else
    proactive_sync_status=$?
  fi
  proactive_sync_reaped_ns=$(date +%s%N)
  proactive_sync_pid=""
  rm -f "$proactive_stop_file"
  summarize_proactive_sync
  write_proactive_lifecycle
}

cleanup_background_processes() {
  stop_proactive_sync
  stop_monitors
}
trap cleanup_background_processes EXIT

java_system_properties=()
java_system_properties+=(
  "-Dcuvs.lucene.experimentalPostIngestOverlap=$post_ingest_overlap"
  "-Dbenchmark.largeVectorWriteExtensions=$large_vector_write_extensions"
)
if [ "$vector_row_copy_mode" = scalar ]; then
  java_system_properties+=("-Dcuvs.benchmark.scalarVectorRowCopy=true")
fi
if [ "$proactive_vector_sync" = true ]; then
  rm -f "$proactive_ready_file" "$proactive_manifest_file" "$proactive_stop_file"
  proactive_sync_spawn_ns=$(date +%s%N)
  if [ "$proactive_sync_helper_mode" = partitioned-overlap-manifest-v1 ]; then
    proactive_sync_args=(
      run-manifest "$proactive_manifest_file" "$proactive_ready_file" "$proactive_stop_file"
      "$proactive_events_file" "$proactive_sync_interval" "$proactive_sync_bin"
    )
  else
    proactive_sync_args=(
      run "$index_dir" "$proactive_ready_file" "$proactive_stop_file" "$proactive_events_file"
      "$proactive_sync_interval" "$proactive_sync_bin"
    )
  fi
  if [ -n "$proactive_sync_start_cutoff_fraction" ]; then
    proactive_sync_args+=(
      "$proactive_sync_expected_payload_bytes" "$proactive_sync_start_cutoff_fraction"
    )
  fi
  "$proactive_sync_helper" "${proactive_sync_args[@]}" \
    > "$results_dir/proactive-vector-sync.log" 2>&1 &
  proactive_sync_pid=$!
  java_system_properties+=("-Dcuvs.proactiveVectorSyncReadyFile=$proactive_ready_file")
  if [ "$proactive_sync_helper_mode" = partitioned-overlap-manifest-v1 ]; then
    java_system_properties+=("-Dcuvs.proactiveVectorSyncManifestFile=$proactive_manifest_file")
  fi
fi

set +e
java_start_ns=$(date +%s%N)
declare -A runtime_native_inode_before=()
declare -A runtime_native_device_before=()
declare -A runtime_native_identity_before=()
declare -A runtime_native_sha256_before=()
for library in "$libcuvs" "$libcuvs_c"; do
  runtime_native_inode_before[$library]=$(stat -Lc '%i' "$library")
  runtime_native_device_before[$library]=$(python3 -c \
    'import os, sys; value = os.stat(sys.argv[1]).st_dev; print(f"{os.major(value):x}:{os.minor(value):02x}")' \
    "$library")
  runtime_native_identity_before[$library]=$(stat -Lc '%d:%i:%s:%Y:%a' "$library")
  runtime_native_sha256_before[$library]=$(sha256sum "$library" | awk '{print $1}')
done
env -u LD_AUDIT -u LD_DEBUG -u LD_DEBUG_OUTPUT -u LD_PROFILE \
  -u LD_PROFILE_OUTPUT -u LD_TRACE_LOADED_OBJECTS \
  LD_LIBRARY_PATH="$native_dir" \
  LD_PRELOAD="$libcuvs:$libcuvs_c" \
  /usr/bin/time -v -o "$results_dir/process-time.txt" \
  java "${BENCHMARK_JAVA_OPTIONS[@]}" \
    "${java_system_properties[@]}" \
    -cp "$runtime_classpath" \
    com.searchscale.lucene.cuvs.benchmarks.LuceneCuvsBenchmarks \
    "$config_file" "$benchmark_id" "$results_dir" \
    > "$results_dir/benchmark.log" 2>&1 &
timed_pid=$!

runtime_native_startup_maps="$results_dir/runtime-native-startup-maps.txt"
runtime_native_maps="$results_dir/runtime-native-maps.txt"
runtime_native_loaded="$results_dir/runtime-native-loaded.tsv"
runtime_native_environ="$results_dir/runtime-native-environ.txt"
runtime_startup_dependencies_loaded="$results_dir/runtime-startup-dependencies-loaded.tsv"
runtime_shared_objects="$results_dir/runtime-shared-objects.tsv"
runtime_maps_verification="$results_dir/runtime-maps-verification.json"
runtime_map_capture_status=missing_java_process
runtime_map_capture_phase=missing
java_pid=
java_executable=$(readlink -f "$(command -v java)")
for ((map_attempt = 0; map_attempt < 300; map_attempt++)); do
  candidates=("$timed_pid")
  children_file="/proc/$timed_pid/task/$timed_pid/children"
  if [ -r "$children_file" ]; then
    read -r -a child_pids < "$children_file"
    candidates+=("${child_pids[@]}")
  fi
  for candidate in "${candidates[@]}"; do
    candidate_executable=$(readlink -f "/proc/$candidate/exe" 2>/dev/null)
    if [ "$candidate_executable" = "$java_executable" ]; then
      java_pid=$candidate
      break 2
    fi
  done
  kill -0 "$timed_pid" 2>/dev/null || break
  sleep 0.1
done

if [ -n "$java_pid" ] && cp "/proc/$java_pid/maps" "$runtime_native_startup_maps" 2>/dev/null \
    && { tr '\0' '\n' < "/proc/$java_pid/environ" \
      | awk '/^(LD_[A-Za-z0-9_]*|CUDA_VISIBLE_DEVICES)=/' | LC_ALL=C sort \
      > "$runtime_native_environ"; }; then
  runtime_map_capture_status=captured_startup
  if ! grep -Fxq "LD_PRELOAD=$libcuvs:$libcuvs_c" "$runtime_native_environ" \
      || ! grep -Fxq "LD_LIBRARY_PATH=$native_dir" "$runtime_native_environ" \
      || [ "$(grep -c '^LD_' "$runtime_native_environ")" -ne 2 ]; then
    runtime_map_capture_status=unexpected_native_environment
  fi
  if [ "$expect_hidden_gpu_control" = true ]; then
    grep -Fxq 'CUDA_VISIBLE_DEVICES=-1' "$runtime_native_environ" || \
      runtime_map_capture_status=unexpected_native_environment
  elif grep -q '^CUDA_VISIBLE_DEVICES=' "$runtime_native_environ"; then
    runtime_map_capture_status=unexpected_native_environment
  fi
  if [ "$expect_hidden_gpu_control" = true ]; then
    if cp "$runtime_native_startup_maps" "$runtime_native_maps"; then
      runtime_map_capture_phase=startup_negative_control
    else
      runtime_map_capture_status=map_copy_failed
    fi
  else
    for ((provider_attempt = 0; provider_attempt < 1200; provider_attempt++)); do
      if grep -Fq '<<< cuVS provider and RMM async memory initialized >>>' \
          "$results_dir/benchmark.log" 2>/dev/null; then
        if cp "/proc/$java_pid/maps" "$runtime_native_maps" 2>/dev/null; then
          runtime_map_capture_phase=post_provider_initialization
        else
          runtime_map_capture_status=post_provider_map_copy_failed
        fi
        break
      fi
      kill -0 "$java_pid" 2>/dev/null || break
      sleep 0.1
    done
    if [ "$runtime_map_capture_phase" != post_provider_initialization ] \
        && [ "$runtime_map_capture_status" = captured_startup ]; then
      runtime_map_capture_status=missing_provider_initialization_marker
    fi
  fi
fi

wait "$timed_pid"
java_status=$?
status=$java_status
if [ -f "$runtime_native_maps" ]; then
  printf 'library\tcanonical_path\tdevice_decimal\tmap_device\tinode\tsize\tmtime\tsha256\tmapping_count\n' \
    > "$runtime_native_loaded"
  if [ "$runtime_map_capture_status" = captured_startup ]; then
    runtime_map_capture_status=verified
  fi
  for library in "$libcuvs" "$libcuvs_c"; do
    read -r device inode size mtime < <(stat -Lc '%d %i %s %Y' "$library")
    map_device=${runtime_native_device_before[$library]}
    mapping_count=$(awk -v path="$library" -v device="$map_device" -v inode="$inode" \
      'NF == 6 && $4 == device && $5 == inode && $6 == path { count++ } END { print count + 0 }' \
      "$runtime_native_maps")
    if [ "$mapping_count" -eq 0 ]; then
      runtime_map_capture_status=missing_native_mapping
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$(basename "$library")" "$library" "$device" "$map_device" "$inode" "$size" "$mtime" \
      "${runtime_native_sha256_before[$library]}" "$mapping_count" \
      >> "$runtime_native_loaded"
  done
  if ! python3 "$frozen_native_helper" verify-maps \
      --maps "$runtime_native_maps" \
      --native-dir "$native_dir" \
      --artifact-dir "$native_artifact_dir" \
      --startup-output "$runtime_startup_dependencies_loaded" \
      --shared-output "$runtime_shared_objects" \
      > "$runtime_maps_verification" 2> "$results_dir/runtime-maps-verification.stderr"; then
    runtime_map_capture_status=map_dependency_verification_failed
  elif [ "$expect_hidden_gpu_control" = false ]; then
    if ! awk -F '\t' 'NR > 1 && $NF == "cuda_driver" { found=1 } END { exit !found }' \
        "$runtime_shared_objects"; then
      runtime_map_capture_status=missing_cuda_driver_mapping
    elif ! awk -F '\t' 'NR > 1 && $NF == "cuda_runtime" { found=1 } END { exit !found }' \
        "$runtime_shared_objects"; then
      runtime_map_capture_status=missing_cuda_runtime_mapping
    fi
  fi
fi
for library in "$libcuvs" "$libcuvs_c"; do
  identity_after=$(stat -Lc '%d:%i:%s:%Y:%a' "$library")
  sha256_after=$(sha256sum "$library" | awk '{print $1}')
  if [ "$identity_after" != "${runtime_native_identity_before[$library]}" ] \
      || [ "$sha256_after" != "${runtime_native_sha256_before[$library]}" ]; then
    runtime_map_capture_status=native_changed_after_launch
  fi
done
if [ "$runtime_map_capture_status" != verified ] && [ "$status" -eq 0 ]; then
  status=11
fi
java_end_ns=$(date +%s%N)
set -e

{
  echo "runtime_java_pid=${java_pid:-not-captured}"
  echo "runtime_java_executable=$java_executable"
  echo "runtime_native_map_capture_status=$runtime_map_capture_status"
  echo "runtime_native_map_capture_phase=$runtime_map_capture_phase"
  if [ -f "$runtime_native_environ" ]; then
    echo "runtime_native_environ_sha256=$(sha256sum "$runtime_native_environ" | awk '{print $1}')"
  else
    echo "runtime_native_environ_sha256=missing"
  fi
  if [ -f "$runtime_native_maps" ]; then
    echo "runtime_native_maps_sha256=$(sha256sum "$runtime_native_maps" | awk '{print $1}')"
  else
    echo "runtime_native_maps_sha256=missing"
  fi
  if [ -f "$runtime_native_startup_maps" ]; then
    echo "runtime_native_startup_maps_sha256=$(sha256sum "$runtime_native_startup_maps" | awk '{print $1}')"
  else
    echo "runtime_native_startup_maps_sha256=missing"
  fi
  if [ -f "$runtime_native_loaded" ]; then
    echo "runtime_native_loaded_sha256=$(sha256sum "$runtime_native_loaded" | awk '{print $1}')"
  else
    echo "runtime_native_loaded_sha256=missing"
  fi
  if [ -f "$runtime_startup_dependencies_loaded" ]; then
    echo "runtime_startup_dependencies_loaded_sha256=$(sha256sum "$runtime_startup_dependencies_loaded" | awk '{print $1}')"
  else
    echo "runtime_startup_dependencies_loaded_sha256=missing"
  fi
  if [ -f "$runtime_shared_objects" ]; then
    echo "runtime_shared_objects_sha256=$(sha256sum "$runtime_shared_objects" | awk '{print $1}')"
  else
    echo "runtime_shared_objects_sha256=missing"
  fi
  if [ -f "$runtime_maps_verification" ]; then
    echo "runtime_maps_verification_sha256=$(sha256sum "$runtime_maps_verification" | awk '{print $1}')"
  else
    echo "runtime_maps_verification_sha256=missing"
  fi
} >> "$results_dir/provenance.txt"

stop_proactive_sync
stop_monitors
trap - EXIT
if [ "$status" -eq 0 ] \
  && [ "$proactive_vector_sync" = true ] \
  && [ "$proactive_sync_status" -ne 0 ]; then
  echo "Proactive vector sync failed with status $proactive_sync_status" >&2
  status=7
fi
record_cache_state "$results_dir/source-cache-after.txt"
find "$index_dir" -maxdepth 1 -type f -printf '%f\t%s\n' | sort \
  > "$results_dir/index-files.tsv"
df -B1 "$dataset_file" "$index_dir" > "$results_dir/filesystems-after.txt"
stat -f "$dataset_file" "$index_dir" > "$results_dir/filesystem-stats-after.txt"
echo "finished_at=$(date --iso-8601=seconds)" >> "$results_dir/provenance.txt"
echo "java_exit_code=$java_status" >> "$results_dir/provenance.txt"
if [ "$proactive_vector_sync" = true ]; then
  echo "proactive_vector_sync_exit_code=$proactive_sync_status" >> "$results_dir/provenance.txt"
  if [ "$proactive_sync_helper_mode" = partitioned-overlap-manifest-v1 ]; then
    if [ -f "$proactive_manifest_file" ]; then
      echo "proactive_vector_sync_manifest_sha256=$(sha256sum "$proactive_manifest_file" | awk '{print $1}')" \
        >> "$results_dir/provenance.txt"
    else
      echo "proactive_vector_sync_manifest_sha256=missing" >> "$results_dir/provenance.txt"
    fi
  fi
fi
echo "exit_code=$status" >> "$results_dir/provenance.txt"

if [ "${CLEAN_INDEX_AFTER_RUN:-true}" = true ]; then
  case "$(realpath -m "$index_dir")/" in
    "$cleanup_root"/*/)
      find "$index_dir" -depth -mindepth 1 -delete
      rmdir "$index_dir"
      ;;
    *)
      echo "Refusing to clean index outside $cleanup_root: $index_dir" >&2
      exit 6
      ;;
  esac
fi

exit "$status"
