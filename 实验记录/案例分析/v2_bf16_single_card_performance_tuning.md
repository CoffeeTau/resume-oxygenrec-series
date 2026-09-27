# OxygenREC-v2 BF16 单卡性能调优分析

## 1. 证据入口

- [batch 64→4096 服务器结果摘录](../案例原始记录/performance_tuning/2026-09-21-v2-bf16-batch-sweep.md)
- [NPU Profiler API 与设备状态预检查](../案例原始记录/performance_tuning/2026-09-21-npu-profiler-precheck.md)
- [NPU batch 4096短窗口Profile](../案例原始记录/performance_tuning/2026-09-21-npu-bs4096-profile.md)
- [NPU batch 4096 Profile热点结果](../案例原始记录/performance_tuning/2026-09-23-npu-bs4096-profile-hotspots.md)
- 服务器原始 JSON 仍保存在 `checkpoints/performance_baseline/` 对应实验目录。

## 2. 判断演变与试错价值

### 初始 64/128/256 结果不能直接归因于模型或 NPU 算子

最初 NPU 吞吐只有 GPU 的约 `43.8%–47.6%`。当时的协议只有20步 warmup，且 batch 64
的 NPU step 时延约66 ms。受控复核把 warmup 提高到100步、测量提高到300步后，NPU batch 64
变为 `3459.94 samples/s`，反而是 GPU 的 `1.506×`。这说明旧结果主要暴露了预热、运行状态
或短窗口问题，不能据此修改模型结构。

### 扩大 batch 是有效的基础摸底步骤

从 batch 128 到1024，两端吞吐持续增长，allocated memory 仍远低于设备容量。NPU/GPU 吞吐比
从约 `1.35×` 收敛到约 `1.24×`，说明固定下发/同步开销随 batch 增大被摊薄。这个过程证明
此前先摸 batch 曲线、再开 Profiler 的顺序是合理的。

### batch 2048 的第一次结果是有价值的失败实验

30步测量窗口下 GPU CV 达 `19.92%`，导致表面 NPU/GPU 比值达到 `1.589×`。该点不是“最好
结果”，而是测量协议失稳。它直接推动了 `V2_PERF_CYCLE_SAMPLES=1` 和100步长窗口复核；复核后
GPU batch 2048 CV 降为 `0.329%`，比值回到 `1.405×`。失败点应保留，避免以后重复把短窗口
抖动解释成硬件收益。

### 吞吐膝点位于 2048–4096 区间

- GPU：1024→2048 吞吐约增加 `4.0%`，2048→4096 约增加 `14.4%`；
- NPU：1024→2048 约增加 `18.0%`，2048→4096 约增加 `12.5%`；
- batch 4096 获得当前最高吞吐，但 GPU CV 为 `6.63%`，高于 batch 2048；
- batch 2048 时延更低且两端更稳定，batch 4096 更适合观察峰值吞吐路径。

因此不再继续盲目扩大 batch。下一阶段在 batch 4096 上采集少量稳态 step 的 Profiler，定位
主机下发、H2D/Memcpy、format/TransData、CPU fallback、loss 主机读取同步、masked_fill 与 AdamW
等候选瓶颈；batch 2048保留为稳定对照点。

## 3. 当前环境风险

Profiler API 已具备，但 `npu-smi` 显示所有设备均非 `OK`，设备6为 `Critical`，设备0为
`Warning` 且在 Util=0 时仍占用约48.4 GB HBM。官方定义中 Warning/Critical 分别代表一般告警
和紧急告警，因此在取得健康 Error Code、usages 和进程信息前，不能把该环境标记为健康。

这不自动否定此前结果：此前稳定 repeat 的 CV 较低，且设备0并非 Critical。健康专项检查不再
阻塞性能主线；正式 Profile 只需同时归档采集前后设备概览。只有设备0报错、存在其他活跃进程、
吞吐显著漂移或采集失败时，才下钻Error Code与usages。异常Profile只作为诊断样本，不进入
优化前后量化比较。

## 4. 后续每轮记录要求

每轮性能实验必须同时保存：

1. 带日期的服务器命令文档；
2. Git commit、tracked worktree 状态及关键源码/input SHA-256；
3. 完整原始 JSON 或 Profiler 输出目录，不只保存终端最后一行；
4. 开始前与结束后的 `npu-smi info`；
5. batch、warmup、measured steps、repeat、精度和设备；
6. 吞吐、step 时延、CV、allocated memory 以及失败/中断信息；
7. 本轮发现、判断、采取的处理、结论边界和下一步；
8. 若废弃某个结果，保留结果并明确废弃原因，不删除失败记录。

## 5. 首轮Profile判断

batch 4096短窗口Profile已成功生成operator、kernel、step trace和timeline产物，训练loss有限，
代码与输入指纹完整。摘要中的`684.56 samples/s`和约`5.98 s/step`包含Profiler采集、同步和解析
开销，不能与无Profiler基线`36859.30 samples/s`比较，也不代表性能发生回退。

E141的原始CSV误删后，E143按相同协议重新采集成功。真实Top operator显示`aclnnDropoutV3`
占24.13%、`aclnnInplaceCopy`占15.93%，FlashAttention反向/正向分别占7.44%/4.99%。kernel层面，
dropout族合计约26.05%，copy/layout族约15.77%，FlashAttention正反向及transpose约12.53%。

此外，operator关键词汇总出现534次`_local_scalar_dense`，即3个active step约178次/step。它的
Device Self Duration为0，因此不能把534次直接换算成总耗时；但结合原生AdamW的状态更新和大量
InplaceCopy/InplaceAdd，它是高优先级host-bound候选。

## 6. 第一项优化决策

不从关闭dropout开始：尽管dropout是最大设备热点，关闭它会改变训练正则化语义，使结果不再是
同一workload的公平性能比较。也不单独开启AdamW `capturable`：当前没有图捕获，PyTorch文档明确
提示该选项可能损害未捕获执行的性能。

首个A/B只把`torch.optim.AdamW`替换为`torch_npu.optim.NpuFusedAdamW`。昇腾的模型优化材料把
融合AdamW列为直接替换项，且TorchNPU 2.7.1分支存在对应接口与测试。该实验保持checkpoint、
模型、dropout、batch、样本顺序、BF16和测量窗口一致，目标是同时减少优化器标量状态、原地更新
和copy开销。

决策门槛为：loss有限、两组CV优先低于5%，且融合版本吞吐至少提高5%。`first_loss`是warmup后的
首个测量loss，不是第一次optimizer更新前的loss，因此只作为数值轨迹诊断，不要求两组一致。若
收益不足或实现不兼容，完整保留负结果，下一轮转向padding mask与layout copy；不重复当前Profile，
也不围绕Level1/2或memory warning继续做细枝末节验证。

## 7. 融合AdamW首轮失败与协议修订

首轮control在server-118取得`27372.037 samples/s`中位吞吐和约`1.22%` CV；用户确认118和119
机器几乎完全一致，因此该点可与历史数据作环境相近的横向参考。但treatment在第一个warmup step
执行`zero_grad(set_to_none=True)`时被TorchNPU拒绝，尚未进入前向、反向或正式测量。

这次失败没有否定融合AdamW性能方向，只说明旧benchmark隐含了原生AdamW支持、融合优化器不支持
的接口前提。若仅把treatment改为`set_to_none=False`，优化器实现和梯度清零策略会同时变化，不能
归因。因此重试必须让两组统一使用zero模式，并把该模式写入结果JSON。旧control继续作为失败流程
记录，不进入新一轮数值比较。

## 8. 融合AdamW A/B结论

匹配zero-grad后的实验运行成功：融合版本中位吞吐从`27150.77`提升到`27879.62 samples/s`，
提高`2.6844%`；中位step时延减少约`3.944 ms`。两组CV均低于1%，且treatment最小吞吐仍高于
control最大吞吐，因此这个小幅差异在本轮具有一致性，不宜简单归为随机噪声。

但收益低于预设5%工程门槛，显存变化也只有约`-0.071%`。此外，两种优化器经过100步warmup后
loss轨迹已经分开；这不等于融合实现错误，也不能解释为质量提升，只说明替换并非完全无数值影响。
综合收益、维护成本和数值边界，本项目不采纳融合AdamW，不进行第二次确认。下一步转向现有Profile
中的layout/copy热点，并优先复用已有operator CSV的call stack，而不是重新跑同类Profile。
<<<<<<< HEAD

## 9. 第二项优化决策：NPU私有格式

现有Profile已经给出明确的copy/layout占比，而源码中没有主动设置`allow_internal_format=False`。
TorchNPU官方接口允许在tensor创建前显式控制是否允许私有格式；因此下一轮不再重复采集Profile，
而是在同一AdamW训练协议中直接比较`disable`与`enable`。这比根据warning猜测某个`masked_fill`源码
位置更快，也能直接回答该layout开关对端到端吞吐是否有工程价值。

本轮唯一变量是私有格式开关：batch 4096、BF16、AdamW、`set_to_none`、数据顺序、100步warmup、
100步测量和3次repeat全部匹配。预设性能门槛仍为5%。由于官方说明开启私有格式后某些算子可能
出现数值路径差异，loss有限只是最低门槛；若性能达到门槛，还必须做第二次性能确认及固定验证集
数值复核，不能直接改正式训练默认值。
=======
>>>>>>> 0daa002 (update)
