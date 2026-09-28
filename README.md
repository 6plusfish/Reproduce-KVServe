# Reproduce KVServe

This repository documents a small reproduction of KVServe using Qwen2.5-7B-Instruct and two NVIDIA A40 GPU nodes. The purpose is to gain practical experience with disaggregated large language model (LLM) serving and understand how KV-cache compression affects communication volume and latency.

## Original Project

- **Project:** KVServe
- **Original repository:** [hpdps-group/KVServe](https://github.com/hpdps-group/KVServe)
- **Paper:** *KVServe: Service-Aware KV Cache Compression for Communication-Efficient Disaggregated LLM Serving*
- **Authors:** Zedong Liu, Xinyang Ma, Dejun Luo, Hairui Zhao, Bing Lu, Wenjing Huang, Yida Gu, Xingchen Liu, Zheng Wei, Jinyang Liu, Dingwen Tao, and Guangming Tan
- **Conference:** ACM SIGCOMM 2026

KVServe is a service-aware KV-cache compression framework for disaggregated LLM serving. In prefill/decode (P/D) disaggregation, a prefill worker computes a prompt's KV cache and transfers it to a decode worker. KVServe reduces this overhead through a modular pipeline combining transformation, quantization, and lossless coding. Its full design also includes offline profiling and an online controller that selects compression profiles according to bandwidth, latency objectives, and quality constraints.

## Reproduction Process

This reproduction validated the main cross-node transfer and compression pipeline:

- P/D disaggregation across two physical GPU nodes
- Cross-node NCCL Socket communication
- Stable request matching through `transfer_id`
- Raw, losslessly compressed, and 8-bit quantized KV transfer
- Short prompt functional tests
- A bandwidth-constrained test with approximately 384-token prompts

This small scale reproduction successfully validated the core KVServe pipeline and provided practical insights into cross node KV cache transfer and compression. However, a complete reproduction would require additional experiments and evaluation.

### Experimental Setup

The experiment used two NVIDIA A40 nodes running Qwen2.5-7B-Instruct. They communicated through the RunPod `podnet1` interface using NCCL Socket.

The same environment and model path were used on both nodes:

~~~bash
source /workspace/kvserve-cu129/bin/activate

hf download Qwen/Qwen2.5-7B-Instruct \
  --local-dir /workspace/models/Qwen2.5-7B-Instruct
~~~

NCCL was configured to use TCP communication:

~~~bash
export NCCL_NET=Socket
export NCCL_SOCKET_IFNAME=podnet1
export NCCL_SOCKET_FAMILY=AF_INET
export NCCL_IB_DISABLE=1
~~~

A two-rank NCCL `all_reduce` test returned `3.0` on both nodes, confirming successful cross-node communication.

### Experiment Configuration

Three requests were tested using short prompts and approximately 384-token prompts.

| Mode | Description |
|---|---|
| `none` | Uncompressed KV transfer |
| `codec-only` | Lossless nvCOMP ANS compression |
| `default` | Default hybrid quantization |
| `HighQ` | 8-bit quantization with nvCOMP ANS |

The successful HighQ configuration used 8-bit min-max quantization, `hybrid_ratio=0.5`, and nvCOMP ANS coding. A ratio of `0.0` was avoided because it caused an empty quantization group and an NCCL communication mismatch.
## Reproduction Results

### Short-Prompt Functional Test

| Mode | Prefill + KV Send | Decode | Output Quality |
|---|---:|---:|---|
| None | 1.554 s | 0.630 s | Correct, 3/3 |
| Codec-only | 1.580 s | approximately 0.71 s | Correct, 3/3 |
| Default | 1.648 s | approximately 0.79–1.00 s | Semantically corrupted |
| HighQ | 1.848 s | 0.832 s | Correct, 3/3 |

Codec-only reduced the payload from 2,752,512 to 1,425,148 bytes, approximately 1.93× lossless compression. HighQ reduced it to 1,080,634 bytes, approximately 2.55× compression and 60.7% fewer transmitted bytes. For short prompts, compression overhead exceeded the communication time saved.

### 384-Token Bandwidth Experiment

| Metric | None | HighQ | Improvement |
|---|---:|---:|---:|
| Total KV bytes | 66,060,288 | 32,256,442 | 51.2% reduction |
| Compression ratio | 1.00× | 2.05× | — |
| Prefill + KV send | 5.051 s | 3.468 s | 31.3% lower; 1.46× speedup |
| Decode | 2.297 s | 1.435 s | 37.5% lower; 1.60× speedup |
| Sum of measured phases | 7.348 s | 4.903 s | 33.3% lower; approximately 1.50× speedup |

All three HighQ outputs remained semantically correct:

- France → Paris
- Machine learning → artificial intelligence
- First human on the Moon → Neil Armstrong

## Result Discussion

The results support KVServe’s central idea on a small scale: reducing KV cache traffic can improve the latency of disaggregated LLM serving when KV transfer accounts for a significant portion of the communication time.

The short prompt test did not benefit from quantization because the amount of transferred KV data was relatively small. In this case, the additional time required for compression and decompression was greater than the communication time saved. In contrast, the 384 token prompts generated a much larger KV cache, allowing compression to provide a clear performance benefit. HighQ reduced KV traffic by 51.2% and lowered the total measured time from 7.348 to 4.903 seconds, representing an improvement of approximately 33.3%.

The default quantization profile produced incorrect outputs with Qwen2.5-7B and vLLM 0.30, while the corrected HighQ configuration preserved the meaning of all three generated outputs. This indicates that compression effectiveness should not be evaluated only by compression ratio or latency. KV cache layout compatibility and generation quality must also be carefully verified.

Overall, the experiment demonstrates that KV cache compression is more effective for longer prompts and bandwidth constrained networks. However, these measurements should not be directly compared with the paper’s main results because the network environment, model, prompt length, request count, compression settings, and evaluation methodology were different.

## Limitations

Only one model, two GPU nodes, three requests, and one constrained network path were tested. Each configuration was measured once, and correctness was checked manually. Controller mode and service aware online selection were not evaluated. The results are a functional and small scale performance validation rather than a complete reproduction of the paper.

Future work should repeat each experiment three to five times, report the mean and standard deviation, test more prompt lengths and bandwidth limits, evaluate quality on a standard dataset, and reproduce the online controller.

## Conclusion

This reproduction completed the main KVServe pipeline across two physical A40 nodes, validating P/D separation, cross-node NCCL KV transfer, lossless and quantized compression, and correct Qwen2.5-7B-Instruct generation.

On an approximately 92 Mbps link, HighQ reduced KV traffic by 51.2%, prefill plus transfer time by 31.3%, and decode time by 37.5% while preserving all three outputs. This confirms the practical value of KV compression under constrained bandwidth, while showing that overhead and numerical quality must be considered together.

## Citation

```bibtex
@article{liu2026kvserve,
  title={KVServe: Service-Aware KV Cache Compression for Communication-Efficient Disaggregated LLM Serving},
  author={Liu, Zedong and Ma, Xinyang and Luo, Dejun and Zhao, Hairui and Lu, Bing and Huang, Wenjing and Gu, Yida and Liu, Xingchen and Wei, Zheng and Liu, Jinyang and Tao, Dingwen and Tan, Guangming},
  journal={arXiv preprint arXiv:2605.13734},
  year={2026}
}
```

## References

1. Liu, Z., Ma, X., Luo, D., Zhao, H., Lu, B., Huang, W., Gu, Y., Liu, X., Wei, Z., Liu, J., Tao, D., and Tan, G. *KVServe: Service-Aware KV Cache Compression for Communication-Efficient Disaggregated LLM Serving*. arXiv:2605.13734, 2026.
2. [Original KVServe implementation](https://github.com/hpdps-group/KVServe)
3. [ACM Digital Library](https://dl.acm.org/doi/10.1145/3789240.3829139)
