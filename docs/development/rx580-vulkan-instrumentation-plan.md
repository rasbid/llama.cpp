# RX580 Vulkan Instrumentation Plan

This document expands the instrumentation portion of the optimization plan for the AMD RX580 (GCN gfx803) Vulkan backend path in `ggml`. The focus is on capturing detailed GPU timings and pipeline statistics to identify the prefill (`mul_mat*`) and infill (`mul_mat_vec*`) bottlenecks.

## Goals

1. Provide accurate per-dispatch GPU timings for all critical matmul pipelines.
2. Collect pipeline statistics when `VK_KHR_pipeline_executable_properties` is supported to understand register, LDS, and instruction usage.
3. Keep instrumentation optional and low-overhead in production builds.
4. Surface the collected data in a format that is easy to analyze.

## Workstreams

### 1. Timestamp Query Instrumentation

1. **Query pool management**
   - Extend the Vulkan context (e.g., `ggml_vk_context`) to own one or more timestamp query pools sized for the maximum number of concurrent dispatches we might record in a command buffer.
   - Ensure pools are created only when instrumentation is enabled, falling back to no-op implementations otherwise.
   - Implement recycling logic so pools can be reset between frames without recreating them.

2. **Command buffer integration**
   - Update `ggml_vk_dispatch_pipeline()` to write timestamps immediately before and after each dispatch when instrumentation is active.
   - Handle command buffers that are pre-recorded vs. dynamically recorded; ensure that timestamp commands are inserted alongside the existing pipeline barrier logic.
   - Guard timestamp writes behind a check for `VK_QUERY_PIPELINE_STATISTIC_COMPUTE_SHADER_INVOCATIONS_BIT` support to avoid validation errors on devices lacking compute timestamp support.

3. **Result retrieval**
   - After submission, collect timestamp results using `vkGetQueryPoolResults()` with `VK_QUERY_RESULT_64_BIT` to maintain precision.
   - Convert timestamp differences to nanoseconds using the device's timestampPeriod.
   - Aggregate results by pipeline name (e.g., `pipeline->name`) and by phase (prefill vs. infill) for easy reporting.

### 2. Pipeline Executable Properties (PEP) Support

1. **Capability detection**
   - During device creation, probe for `VK_KHR_pipeline_executable_properties` and store the support flag in the device capabilities structure.
   - Gate any PEP usage behind this flag so unsupported drivers do not incur additional calls.

2. **Data collection**
   - Add helper routines that call `vkGetPipelineExecutablePropertiesKHR` and `vkGetPipelineExecutableStatisticsKHR` for pipelines that are executed when instrumentation is enabled.
   - Focus on collecting metrics relevant to matmul tuning, such as LDS usage, SGPR/VGPR counts, and instruction counts.
   - Cache the results per pipeline to avoid repeated expensive queries.

3. **Reporting**
   - Integrate PEP data into the same reporting channel as timestamp results, clearly annotating pipelines with their resource usage stats.
   - Provide a summary table in the logs or exported JSON to highlight potential register pressure or occupancy issues specific to the RX580.

### 3. Configuration & UX

1. **Runtime controls**
   - Introduce an environment variable (e.g., `GGML_VK_PROFILING=1`) or a build-time option to toggle instrumentation. Default to disabled.
   - When enabled, log a concise message describing which instrumentation features are active (timestamps, PEP).

2. **Data output**
   - Emit human-readable log lines summarizing per-dispatch timings and pipeline stats.
   - Optionally generate a structured JSON blob that contains:
     ```json
     {
       "device": "AMD Radeon RX580",
       "timestamp_period_ns": <number>,
       "dispatches": [
         {
           "pipeline": "mul_mat_q4_0_l",
           "phase": "prefill",
           "time_us": 123.4,
           "executables": {
             "LDSUsage": "32KB",
             "VGPRs": 64,
             "SGPRs": 96
           }
         }
       ]
     }
     ```
   - Ensure the logging respects existing verbosity settings to avoid flooding standard output during regular runs.

3. **Validation & Testing**
   - Add unit/integration tests in the Vulkan backend (where feasible) to confirm instrumentation paths do not crash when enabled/disabled.
   - Run manual validation on an RX580: execute representative prefill and infill workloads, capture the logs/JSON, and verify that timings are recorded for all relevant pipelines.

## Implementation Checklist

- [x] Add instrumentation configuration flag and device capability storage.
- [x] Create timestamp query pools and wire them into `ggml_vk_dispatch_pipeline()`.
- [x] Implement result aggregation and logging/JSON export.
- [x] Hook up `VK_KHR_pipeline_executable_properties` data collection.
- [x] Document usage instructions for developers profiling the RX580 path.

## Usage

Set `GGML_VK_PROFILING=1` to enable the Vulkan profiler. The backend logs the active features (timestamps and pipeline executable
properties) and prints a per-dispatch breakdown for every `mul_mat*` and `mul_mat_vec*` kernel, followed by an aggregated summary
with the most relevant AMD statistics (VGPRs, SGPRs, LDS usage, etc.). Set `GGML_VK_PROFILING=json` to emit the same information
as a JSON blob in addition to the human-readable log. Disable the environment variable to return to the zero-overhead fast path.

The output contains:

- Individual dispatch timings with workgroup sizes for prefill (`mul_mat*`) and infill (`mul_mat_vec*`) pipelines.
- Aggregated averages and totals grouped by pipeline and phase, annotated with cached pipeline executable statistics when the
  device supports `VK_KHR_pipeline_executable_properties`.
- Optional structured JSON mirroring the log content for downstream analysis.

## Expected Outcomes

- Developers can pinpoint the specific matmul kernels that dominate RX580 runtime, with precise GPU timings.
- Pipeline statistics illuminate whether occupancy, register pressure, or LDS saturation contribute to bottlenecks.
- Instrumentation remains optional, enabling routine builds to stay lightweight while providing deep insights when needed.
