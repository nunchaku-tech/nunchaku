| Batch    | Results                                                     | Inference Time (sec)  |
|---------|------------------------------------------------------------|------------|
| Batch 1 | ![Batch 1](results/batch_1/flux.1-dev.png)            | 27.7s    |
| Batch 2 | ![Batch 2](results/batch_2/flux.1-dev.png)            | 55.0s    |
| Batch 3 | ![Batch 3](results/batch_3/flux.1_dev.png)            | 1m.24s    |



## 3. Potential Overhead from Loop-Based Batch Processing

In some parts of the code, you may see a pattern like:

```cpp
for (int i = 0; i < batch_size; i++) {
    concat.slice(0, i, i + 1).copy_(encoder_hidden_states.slice(0, i, i + 1));
    qkv_proj.forward(...);
    // ...
}
