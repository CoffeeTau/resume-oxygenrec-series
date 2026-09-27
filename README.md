# OxygenREC paper-method reimplementation

This repository is a clean-room PyTorch-oriented reimplementation of
**OxygenREC: An Instruction-Following Generative Framework for E-commerce
Recommendation**.

The target is not a strict reproduction of JD.com's production system. The
paper's private data, feature definitions, checkpoints, reward service, and
serving stack are unavailable. Results produced here must be described as:

> OxygenREC paper-method reimplementation on a public-data approximate benchmark.

This is also a transition project toward general LLM and Agentic Search work.
Implementation and review prioritize transferable reasoning, retrieval,
semantic alignment, policy optimization, and trajectory methods. Recommendation-
specific SID tuning and industrial serving receive only enough work to validate
their GPU control flow.

## Current milestone

完整项目复盘建议先看 [OxygenREC项目复盘与面试指南](OxygenREC项目复盘与面试指南.md)，
它按数据、模型、算法、测评、调优串起 v1/v2、GPU/NPU 和实验边界。只想阅读 v1
源码调用链时，再看 [代码阅读与数据流指南](代码阅读与数据流指南.md)。
当前用于简历质量增益验证的收敛主线、服务器命令与结果路径见
[V1 Fast-Slow / Semantic ID / IGR-Q2I 实验主线](V1_FastSlow_SemanticID_IGR_Q2I实验主线.md)。

Phase 1 starts with the smallest auditable loop:

```text
behavior log -> temporal split -> item SID codebook -> history SIDs
             -> encoder-decoder -> constrained SID decoding
             -> item IDs -> HR/Recall
```

Implemented now:

- an immutable three-level Semantic ID value object;
- a versioned item-to-SID registry with collision reporting;
- a prefix trie for legal constrained decoding;
- a dataset-neutral interaction schema and streaming RetailRocket adapter;
- global temporal boundaries and leak-resistant next-item sample construction;
- a deterministic residual K-means reference and SID quality diagnostics;
- a small dense Transformer encoder-decoder with level-aware SID embeddings;
- three level-specific prediction heads and weighted next-token loss;
- greedy PrefixTrie-constrained Semantic-ID decoding;
- deterministic reference beam search and HR/Recall/MRR/NDCG evaluation;
- contextual instruction fusion and Q2I semantic alignment;
- long-history IGR and bounded Qwen Retrieval Plan controls;
- local Qwen3-4B structured reasoning generation on GPU;
- public-proxy SA-GCPO objectives and rollout validation;
- dependency-free unit tests for these invariants.

Current handoff:

1. GPU-side OxygenREC-v1 and v2 method reproduction is closed within the
   public-proxy scope; this is not a stable quality or private-table claim;
2. NPU Stage-0 passed on an 8-card Ascend 950DT server: TorchNPU/HCCL are
   available and one-card tensor, backward, optimizer, and checkpoint checks
   succeeded;
3. the v2 Full GPU/NPU FP32 and BF16 20-step training/checkpoint reload gates
   pass from the same commits and input hashes; fixed validation also passed
   input, finite-value, legality, mean-loss, and aggregate-metric gates. FP32
   target/greedy/beam fingerprints match, while BF16 target/greedy match but
   beam differs; this is an optional diagnostic, not a blocker for full-epoch
   training on the same public-data proxy;
4. MoE and production-serving optimization remain deferred.

See [the reuse survey](docs/reference_reuse.md) and
[explicit implementation assumptions](configs/assumptions.yaml). The current
model shapes, masks, loss, and validation commands are in
[the Phase-1 model protocol](docs/model_protocol.md). The bounded real-data run
is documented in [the training protocol](docs/training_protocol.md).
中文总体进度见 [复现进度](复现进度.md)，逐次实验判断与修正过程见
[复现实验日志](实验记录/复现实验日志.md)。
GPU→NPU的分阶段门槛见[NPU迁移计划](docs/npu_migration_plan.md)。
GPU与NPU服务器环境分别见[环境快照索引](docs/server_environment.md)。
后续需要在服务器执行的可复制命令统一见
[服务器执行命令索引](docs/server_commands/README.md)，不要再从早期的
`tmp_command.txt` 或聊天记录复制。
服务器直接下载并接入Qwen的逐步操作见
[Qwen服务器接入操作指南](Qwen服务器接入操作指南.md)。

## Run the current tests

The initial SID layer uses only the Python standard library:

```bash
python3 -m unittest discover -s tests -v
```
