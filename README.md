# Performance Benchmarks

This document presents the performance comparison between the Naive and FB_cache implementations.

## 1. Naive Implementation

| Batch    | Results                                               | Inference Time (sec) |
|----------|-------------------------------------------------------|----------------------|
| Batch 1  | ![Batch 1](results/batch_1/flux.1-dev.png)            | 27.7s                |
| Batch 2  | ![Batch 2](results/batch_2/flux.1-dev.png)            | 55.0s                |
| Batch 3  | ![Batch 3](results/batch_3/flux.1-dev.png)            | 1m 24s               |

## 2. FB_cache Implementation

| Batch    | Results                                               | Inference Time (sec) |
|----------|-------------------------------------------------------|----------------------|
| Batch 1  | ![Batch 1](results/batch_1_FB_cache/flux_t_batch_1.png) | 9.0s                 |
| Batch 2  | ![Batch 2](results/batch_2_FB_cache/flux_t_batch_2.png) | 19.0s                |
| Batch 3  | ![Batch 3](results/batch_3_FB_cache/flux_t_batch_3.png) | 29.2s                |

## 3. Potential Overhead from Loop-Based Batch Processing

In some parts of the code, a loop-based batch processing pattern is used. This approach can introduce overhead when processing large batches, and may require optimization. For example:

```cpp
for (int i = 0; i < batch_size; i++) {
    concat.slice(0, i, i + 1).copy_(encoder_hidden_states.slice(0, i, i + 1));
    qkv_proj.forward(...);
    // additional operations...
}
