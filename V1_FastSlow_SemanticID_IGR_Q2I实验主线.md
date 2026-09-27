> 这条主线只研究 OxygenREC-v1：Semantic ID、Fast-Slow Thinking、Q2I 和 IGR。v2、RL、MoE 与 NPU 性能不进入本轮质量实验。

# 1. 项目现在要证明什么

一句话目标：在 RetailRocket 公开数据代理上，先用稳定的 Semantic ID 表示商品，再让冻结的 Qwen3-4B-Instruct-2507 离线理解用户历史，最后验证 Slow Instruction、Q2I 对齐和 IGR 长历史检索能否提升 Fast 生成式推荐的 HR@5、MRR 与 NDCG。

这里不是“把论文模块都跑一遍”，而是验证三条递进假设：

1. **Semantic ID 是否可用**：商品向量能否被三层残差量化，并保持可接受的重构误差、碰撞率和合法解码率；
2. **Slow Instruction 是否有效**：Qwen 从 target 之前的历史生成意图特征后，Fast 模型是否优于只看短历史的 Base；
3. **Q2I / IGR 是否继续带来收益**：Q2I 是否改善 query-target cosine，IGR 是否从较早历史中找回真实重复商品，并最终反映到推荐指标。

项目的诚实边界是：这是 OxygenREC 方法在公开数据上的代理实验，不是京东私有数据、线上系统或论文主表复现。

# 2. 整体链路

```text
RetailRocket events.csv
  │
  ├─ 全局时间切分 80% / 10% / 10%
  │    train / validation / test
  │
  ├─ 训练期商品属性 → 256维公开代理向量
  │    → 3层 RQ-KMeans → Semantic ID Registry
  │
  └─ target 之前的用户历史
       ├─ 最近20条 → Fast Encoder 主输入
       ├─ 更早100条 → IGR 候选池
       └─ 行为统计 + 最近/重复 SID 锚点
            → 冻结 Qwen 离线生成 Instruction
            → 冻结 Qwen hidden state 缓存
                 │
                 ├─ Q2I：Instruction query 对齐目标 item
                 └─ IGR：query 从长历史选择 Top-10
                        ↓
              Fast Transformer Encoder-Decoder
                        ↓
              PrefixTrie 约束 Beam Search
                        ↓
       HR@5 / MRR / NDCG / Legal SID / Q2I cosine / IGR recall
```

Fast-Slow 的“Slow”不是每次在线请求都调用大模型。本实验先离线生成并缓存 Qwen 特征；训练和推荐阶段只读取缓存、训练小型 Fast 模型。这样既接近近线/在线分工，也避免四组消融重复运行 4B 模型。

# 3. 数据和防泄漏设计

## 3.1 时间切分

事件使用全局时间边界切成 train、validation、test。对时刻 $t$ 的 target，history 只能包含同一用户、严格早于 $t$ 的事件；同毫秒事件互不可见。

代码入口：

- [`src/oxygenrec/data/temporal.py`](src/oxygenrec/data/temporal.py)：时间切分、下一商品样本、确定性 reservoir sampling；
- [`src/oxygenrec/data/model_inputs.py`](src/oxygenrec/data/model_inputs.py)：短历史/长历史分离和张量补齐；
- [`docs/data_protocol.md`](docs/data_protocol.md)：完整数据边界。

## 3.2 固定 cohort，拆开两类随机性

过去 `--seed` 同时控制“抽哪些样本”和“模型怎样初始化”，多随机种子对比会换样本，无法严格归因。现在拆成：

- `--sample-seed 17`：所有实验固定同一 train/validation/test cohort；
- `--seed 17/23/42`：只改变模型初始化与训练 shuffle。

四个变体还统一启用 `--matched-igr-cohort`，因此每条样本都有至少 20 条短历史和 10 条可检索的较早历史。

## 3.3 Slow 输入不读取 target

Qwen 只能看到 target 之前的：

- 历史长度和 view/cart/transaction 计数；
- 最近行为序列；
- 最近 12 个商品的匿名 SID 锚点；
- 最多 6 个重复商品 SID 及次数。

不输入 target 商品、target 行为或用户 ID。SID 锚点解决了旧实验“只有行为统计，没有商品信号”的问题，但它仍不等价于真实标题、类目文本或多模态商品语义。

对应代码：[`scripts/cache_qwen_instructions_retailrocket.py`](scripts/cache_qwen_instructions_retailrocket.py)。

# 4. 三个核心模块怎样工作

Fast-Slow、Q2I 与 IGR 的原始框架来自 [OxygenREC 论文](<Hao 等 - 2025 - OxygenREC An Instruction-Following Generative Framework for E-commerce Recommendation.pdf>)。下面公式按当前公开代理实现展开；其中行为日志、SID 文本锚点、方差/去相关正则和 exact-item 诊断属于本项目的可审计实现，不冒充论文私有生产细节。

## 4.1 Semantic ID

对商品代理向量 $x$ 做三层残差量化。第 $l$ 层选择最接近当前残差的码字：

$$
c_l=\arg\min_j\left\|r_{l-1}-e_{l,j}\right\|_2^2,
\qquad
r_l=r_{l-1}-e_{l,c_l},
\qquad r_0=x.
$$

最终商品被表示成三元组：

$$
\operatorname{SID}(x)=(c_1,c_2,c_3).
$$

当前冻结配置是 width=256、K-Means++、seed=17。已有记录中，18,733 个商品的重构 MSE 约为 0.001678，colliding-item rate 约为 20.33%。这比早期 width=64 的 91.16% 碰撞显著改善，但 20.33% 仍是公开代理限制。

因此本轮同时报告：

- 推荐指标：按当前 registry 的 SID/item 映射计算；
- IGR SID recall：与原实现兼容；
- **IGR exact-item recall**：新增主机制指标，防止两个不同商品因 SID 碰撞被误算为召回成功；
- Legal SID Rate：PrefixTrie 约束解码的合法率。

代码入口：[`scripts/fit_retailrocket_sid.py`](scripts/fit_retailrocket_sid.py)、[`src/oxygenrec/quantization_torch.py`](src/oxygenrec/quantization_torch.py)、[`src/oxygenrec/sid.py`](src/oxygenrec/sid.py)。

## 4.2 Fast-Slow Thinking

Slow 侧流程：

```text
严格历史证据
  → Qwen 生成结构化 JSON reasoning
  → intent/evidence/retrieval_strategy + SID anchors
  → Qwen 最后一层 hidden state
  → FP16 特征缓存
```

Fast 侧每个 batch 只加载 `[B, 2560]` 的缓存特征，经可训练 Linear adapter 投影到 Fast hidden size。Instruction 同时进入 Decoder 前缀，并用于构造 Q2I/IGR query。

旧缓存只有 32 train + 32 validation，不能支撑质量结论。新默认缓存覆盖：

- train：2,000；
- validation：500；
- test：500。

规模都可通过环境变量调整，但一次正式对比中不能临时改 cohort。

## 4.3 Q2I

模型把 scenario、Slow Instruction 和最近商品 trigger 融合成 query：

$$
q=\operatorname{norm}\left(f_q([I_s;I_r])\right),
\qquad
z_i=\operatorname{norm}\left(f_i(E(\operatorname{SID}_i))\right).
$$

基础对齐项是负余弦相似度：

$$
\mathcal{L}_{align}=-\frac{1}{B}\sum_{b=1}^{B}q_b^\top z_{i_b}.
$$

当前实现还加入 batch 方差保持与去相关项，避免 query 坍缩；训练目标为：

$$
\mathcal{L}=\mathcal{L}_{NTP}+\lambda\mathcal{L}_{Q2I},
\qquad \lambda=0.2.
$$

新增的只读诊断会让 B 和 C 都输出 validation/test `q2i_cosine`。因此可以直接回答：加入 Q2I loss 后，query-target cosine 是否真的提高，而不是只看最终 HR。

## 4.4 IGR

IGR 对较早历史中每个候选商品计算：

$$
s_j=\cos(q,z_j)=q^\top z_j,
$$

再选择 Top-10 拼到最近 20 条短历史之后。Fast Encoder 实际读取 30 个商品，而不是整个 120 条窗口。

IGR 必须同时和两个无需学习的基线比较：

- `recent`：直接取长历史中最近 10 条；
- `random`：从有效长历史均匀无放回抽 10 条的解析期望。

本轮以 exact-item recall 为主，SID recall 只作兼容诊断。IGR 若没有超过 recent/random，就不能写成正向收益；此时最终项目可以保留 Fast-Slow+Q2I，把 IGR 写成 bad case 分析和后续优化。

# 5. A/B/C/D 实验设计

| 组别 | `--variant` | Slow Qwen | Q2I loss | IGR | 回答的问题 |
|---|---|---:|---:|---:|---|
| A | `base` | 否 | 否 | 否 | 仅 Fast 短历史能做到什么 |
| B | `qwen_instruction` | 是 | 否 | 否 | Slow Instruction 自身是否增益 |
| C | `qwen_q2i` | 是 | 是 | 否 | 对齐约束是否进一步增益 |
| D | `igr_qwen_q2i` | 是 | 是 | 是 | 长历史检索是否继续增益 |

公平性控制：

- 四组使用相同 SID registry、时间边界、sample seed、样本数、batch size、beam width；
- 每个模型 seed 先训练自己的共同 Base checkpoint；
- A/B/C/D 从该 seed 的同一个 Base checkpoint 热启动，再训练相同 epoch；
- B/C/D 新增的 Qwen adapter 在同一 model seed 下使用相同初始化；
- IGR 只对历史位置表做可解释的前缀扩展，其余同形状参数全部加载；
- Qwen 缓存只生成一次，被全部 seed 和 Qwen 变体复用；
- 最佳 epoch 只按 validation 的 HR@5→MRR→NDCG 选择；
- screen 阶段不读取 test，confirm 阶段才对最佳 epoch 评测一次 test。

# 6. 两阶段运行，避免浪费时间

## 6.1 第一阶段：单 seed 筛选

在服务器项目根目录执行：

```bash
export QWEN_MODEL=/你的模型目录/Qwen3-4B-Instruct-2507
export PYTHON_BIN=python
RUN_MODE=screen ./run_v1_fast_slow_experiments.sh
```

默认会自动完成：

1. seed=17、100K 样本的共同 Base 预训练；
2. 2K/500/500 的 Qwen 特征缓存；
3. A/B/C/D 四组相同 cohort 的 8 epoch 微调；
4. validation 结果汇总。

如果服务器路径不同，只覆盖环境变量，不要改脚本：

```bash
EVENTS=/data/retailrocket/events.csv \
SID_REGISTRY=/data/rq/w256_kmeanspp/sid_registry.json \
QWEN_MODEL=/models/Qwen3-4B-Instruct-2507 \
RESULT_ROOT=/data/oxygenrec_runs/v1_fast_slow_mainline \
RUN_MODE=screen \
./run_v1_fast_slow_experiments.sh
```

脚本可断点续跑：存在 `best.pt` 和 `result.json` 的完成任务会跳过；Qwen 缓存每 10 个 batch 保存一次 `*.progress.pt`，中断后使用相同命令会从已校验的样本前缀继续。日志与缓存保留在固定目录。

如果服务器已有同一 SID registry、同一时间边界和同一模型配置的 100K Base，screen 可直接复用：

```bash
BASE_CHECKPOINT_17=/已有目录/epoch-3.pt \
QWEN_MODEL=/models/Qwen3-4B-Instruct-2507 \
RUN_MODE=screen \
./run_v1_fast_slow_experiments.sh
```

代码会再次校验 checkpoint 的 registry version 与时间边界；不匹配时直接失败，不会静默加载。三 seed confirm 也可分别提供 `BASE_CHECKPOINT_17/23/42`。

## 6.2 第二阶段：三 seed 确认

第一阶段截图返回分析后，从 B/C/D 中选一个候选。例如候选是 C：

```bash
export QWEN_MODEL=/你的模型目录/Qwen3-4B-Instruct-2507
RUN_MODE=confirm CONFIRM_VARIANT=qwen_q2i ./run_v1_fast_slow_experiments.sh
```

脚本会运行 seeds 17/23/42 的 Base 与候选方法，并在最佳 validation epoch 上各评测一次 test。

不要根据 test 结果换候选或调参；否则 test 也变成 validation，最终增益不再可信。

## 6.3 首轮 screen 无增益时：运行 V1.1 定向修正

首轮实测出现了两个明确问题：Q2I cosine 上升但推荐列表不变；纯语义 IGR
的 exact-item recall 低于 recent/random。V1.1 只针对这两个问题修改，不重新生成
昂贵的 Qwen 缓存：

- Q2I query 以残差形式进入 Decoder prompt，使辅助表征进入主排序链路；
- 加入 batch 内多正样本对比损失，重复 target 不会被误当成负样本；
- Q2I target 侧停止更新共享 SID embedding，降低辅助损失破坏 NTP 的风险；
- IGR 使用 `0.5 × semantic + 0.5 × recency`，并把检索量从 10 降到 5；
- cohort 仍固定为至少 10 条长历史，因此与首轮 Qwen cache 完全一致。

服务器拉取新代码后直接运行：

```bash
RUN_MODE=refine ./run_v1_fast_slow_experiments.sh
```

它默认复用：

```text
artifacts/v1_fast_slow_mainline/pretrain/seed-17/base/best.pt
artifacts/v1_fast_slow_mainline/cache/qwen_instruction_features.pt
artifacts/v1_fast_slow_mainline/cache/qwen_instruction_reasoning.jsonl
```

新结果单独写到：

```text
artifacts/v1_fast_slow_v11/
```

因此不会覆盖或跳过首轮结果，也不需要再次加载 Qwen3-4B。最后的终端汇总会直接显示
`best_epochs`、`q2i`、`exact_igr`、`recent`、`delta_recent` 和 `random`，只需截图
这几行即可分析。若输出被终端滚走，可执行：

```bash
RUN_MODE=refine_summary ./run_v1_fast_slow_experiments.sh
```

只有候选 HR@5 为正、MRR/NDCG 不同时恶化时才进入确认。例如 V1.1 的 C 胜出：

```bash
RUN_MODE=refine_confirm CONFIRM_VARIANT=qwen_q2i \
  ./run_v1_fast_slow_experiments.sh
```

若 D 胜出，则把候选改成 `igr_qwen_q2i`。此外，D 的
`delta_recent` 应大于 0；否则即使 HR@5 偶然提高，也不能声称 IGR 检索机制优于简单时序基线。

## 6.4 V1.1 仍无增益时：V1.2 残差注入与保守融合

V1.1 的实际 screen 结果为：Base HR@5=`0.358`，Q2I 与 IGR 均为 `0.354`；
IGR 的 exact recall 从首轮约 `0.568` 提高到 `0.585`，但仍低于 recent=`0.610`。
这说明时间融合缩小了检索差距，却还没有形成推荐增益。

继续检查代码后发现，Qwen 变体原先用随机初始化的
`instruction_feature_adapter(Qwen feature)` **完全替换** Base 已训练好的 reasoning prompt。
这会让 Slow 分支在微调开始时发生不必要的分布偏移。V1.2 改为：

```text
reasoning = Base reasoning prompt
          + 0.1 × Linear(LayerNorm(Qwen feature))
```

其中 Linear 零初始化，因此训练第 0 步严格退化为 Base；只有 Slow 特征学到有效梯度后才逐步影响 Decoder。
V1.2 同时移除 Qwen 分支额外添加的 last-item trigger（该信息已存在于 Encoder 历史中），并在模型构造后
重置训练 RNG，避免不同变体因新增模块消耗不同随机数而得到不同的 dropout 序列。
Q2I 同时改为更保守的权重：总权重 `0.02`、Decoder 残差 `0.10`、对比项 `0.5`；
IGR recency 权重提高到 `0.75`，继续减少不可靠语义检索带来的替换。

运行命令：

```bash
RUN_MODE=refine2 ./run_v1_fast_slow_experiments.sh
```

仍复用首轮 Base 与 Qwen cache，新结果写入：

```text
artifacts/v1_fast_slow_v12/
```

V1.2 还会为每个最佳 epoch 保存 `validation_rankings.json`，随后自动执行 Base/Slow
Reciprocal Rank Fusion。终端会额外输出：

```text
FUSION candidate=qwen_instruction ...
FUSION candidate=qwen_q2i ...
FUSION candidate=igr_qwen_q2i ...
```

融合网格包含 `alpha=0`，即纯 Base，因此验证集选择结果不会为了“必须融合”而接受一个低于 Base
的方案。需要同时截图普通 `variant=...` 与 `FUSION ...` 两部分；前者判断单模型增益，后者判断
Fast 主模型与 Slow 分支是否存在互补预测。

# 7. 结果在哪里

默认根目录：

```text
artifacts/v1_fast_slow_mainline/
├── cache/
│   ├── qwen_instruction_features.pt
│   └── qwen_instruction_reasoning.jsonl
├── pretrain/seed-*/base/
│   ├── best.pt
│   └── result.json
├── screen/seed-17/{base,qwen_instruction,qwen_q2i,igr_qwen_q2i}/
│   ├── best.pt
│   ├── metrics.jsonl
│   └── result.json
├── screen/{summary.json,summary.csv,summary.md}
├── confirm/seed-*/{base,候选方法}/
├── confirm/{summary.json,summary.csv,summary.md}
└── logs/*.log
```

请优先截图终端最后这几行，以及打开对应的 `summary.md`：

```text
V1_FAST_SLOW_SUMMARY ...
variant=base ...
variant=qwen_instruction ...
variant=qwen_q2i ...
variant=igr_qwen_q2i ...
SUMMARY_JSON=...
SUMMARY_CSV=...
SUMMARY_MD=...
```

如果终端太长，可随时重新汇总，不重训：

```bash
RUN_MODE=summary ./run_v1_fast_slow_experiments.sh
```

# 8. 怎样判断实验是否足以写进简历

推荐质量的主判断标准：

- 三 seed test 平均 HR@5 相对 Base 提升至少 5%；
- 至少 2/3 seed 的 HR@5 为正增益，另一个不应出现严重反向；
- MRR 与 NDCG 不能同时明显下降；
- Legal SID Rate 应保持 1.0；
- 方法指标要能解释结果：Q2I 看 cosine，IGR 看 exact-item recall 与 recent/random。

结果可能形成三种结论：

1. **B 胜出**：Slow Instruction 已有增益，Q2I/IGR 暂无额外价值；
2. **C 胜出**：Fast-Slow+Q2I 是最适合包装的主线，IGR 作为失败分析；
3. **D 胜出**：可以完整讲“意图理解→query-item 对齐→长历史检索→生成推荐”。

若所有组都没有稳定提升，不应硬写“优化推荐效果”。仍可根据 summary 判断是：Instruction 没提供商品信号、Q2I cosine 提升但没有转成排序、IGR 找回历史但生成器没利用，还是 Fast 模型本身欠训练，然后只补一个最有针对性的实验。

**思考**：为什么不能只比较 D 和 A？

<details>
<summary><strong>参考分析</strong></summary>

因为 D 同时加入 Qwen、Q2I 和 IGR。即使 D 提升，也无法知道收益来自哪个环节；如果 D 下降，也无法定位是哪一层引入噪声。A/B/C/D 的递进消融让每一步只有一个主要变量。

</details>

**思考**：为什么 IGR 要报告 exact-item recall，而不是只报告 SID recall？

<details>
<summary><strong>参考分析</strong></summary>

当前 registry 仍有约 20.33% colliding-item rate。两个不同商品可能共享完整 SID，SID 命中不一定是真实商品命中。exact-item recall 能避免把这种碰撞误报为检索成功。

</details>

# 9. 关键代码阅读顺序

建议按真实调用链阅读：

1. [`run_v1_fast_slow_experiments.sh`](run_v1_fast_slow_experiments.sh)：实验编排、统一参数和产物路径；
2. [`scripts/train_retailrocket.py`](scripts/train_retailrocket.py)：cohort、变体开关、热启动、训练、validation/test；
3. [`scripts/cache_qwen_instructions_retailrocket.py`](scripts/cache_qwen_instructions_retailrocket.py)：无泄漏历史证据和 Qwen 缓存；
4. [`src/oxygenrec/model.py`](src/oxygenrec/model.py)：Instruction、Q2I、IGR、Encoder-Decoder、Beam Search；
5. [`src/oxygenrec/data/model_inputs.py`](src/oxygenrec/data/model_inputs.py)：短/长历史张量如何构造；
6. [`scripts/summarize_v1_fast_slow.py`](scripts/summarize_v1_fast_slow.py)：多 seed、相对增益和机制指标汇总。

# 10. 一页总结

- 数据：全局时间切分，history 严格早于 target，固定 sample seed；
- SID：三层 RQ，冻结 width=256/K-Means++ registry，同时承认 20.33% 碰撞限制；
- Slow：Qwen 只离线运行一次，输入行为证据和匿名历史 SID 锚点；
- Fast：小型 Transformer Encoder-Decoder，PrefixTrie 约束生成合法 SID；
- Q2I：用辅助损失拉近 instruction query 与 target item；
- IGR：从较早 100 条历史中选 Top-10，exact-item recall 对比 recent/random；
- 实验：先 seed17 A/B/C/D 筛选，再 Base vs 最优组做 17/23/42 test；
- 结论：只根据统一汇总中的真实增益决定最终简历写 Fast-Slow、Fast-Slow+Q2I，还是完整 IGR/Q2I 链路。
