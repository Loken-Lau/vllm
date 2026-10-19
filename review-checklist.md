# vLLM pr-checklist 最终复核：未发现阻止发布的问题

执行的 skill：[pr-checklist](../../vllm/.agents/skills/pr-checklist/SKILL.md)。
复核对象：`perf/audio-encoder-compact-mask`，实验基线 `283d48f4ebb404b00e76264ad8f759d7e9a05dfd`。

## 1. Design Fit

两个音频模型的 key padding additive bias 不随 query 位置变化，使用 singleton query 维保留同一组 bias。只移除两处平方 expand；保留浮点语义、dtype/device 和长度公式。没有 scheduler、runner、KV 或 kernel 改动，没有新分派、依赖、配置、helper 或主线程同步。新的 FlashAttention encoder 支持不在此 PR 内。内存收益有实际存储与分配峰值证据，非单纯风格清理。

## 2. Testing and Validation

- 已有 processing 文件新增 4 个参数化用例：真实模型方法 + 真实小 encoder，SimpleNamespace 仅承载 encoder / Identity projector，没有模拟 attention 输出。对照 independently built dense mask，检查实际存储和输出。相关文件 7 passed；原始两源码 + 新用例 4 expected failures。
- 最低 Transformers 5.16.1 与当前 5.19.0 各跑 7 项回归和 96 项补充矩阵；192 encoder comparisons 最大差异 0。覆盖有效奇偶帧数、单/混合 batch、CPU FP32 / GPU 三种 dtype、默认/SDPA/eager。每版本另有六项 mask-builder 边界和负对照。
- A800 / BF16 / S=1500 的 B=1/4/16 mask storage 缩减 1500 倍，SDPA 差异 0。每组五次独立构建计时，保留 IQR，B=1 未获得构建加速。
- 两个官方预训练模型的独立引擎 A/B，80 对测量请求，文本/token/top-5 logprobs 完全一致；35s AF3 输入确实分块。hooks 确认实际 SDPA、平方/紧凑 mask 和存储量。两种实现均通过 GPU sync check。
- 独立进程计时有毛刺及表观退化，因此额外执行同进程 ABBA，每轮每负载预热5次、测量20次。每版本每负载40样本，中位数变化 -0.87% 到 +0.42%，范围重叠，不宣称端到端加速。全部输出与 logprobs 仍一致；未删异常样本。
- test tethering 通过；已有 `.buildkite/test_areas/models_multimodal.yaml` processing suite 会执行新增用例。未运行上游 CI、全仓库测试或其他硬件/TP/量化/encoder编译配置，不声称覆盖这些项。不是 WER 数据集评测。

## 3. Code Quality and Style

适用 pre-commit 全通过，含 Ruff、mypy、SPDX、forbidden imports 和 CI tethering。三份上游文件 SHA-256 与人类审阅时一致；没有额外代码变更。新增测试复用已有文件、参数化复用 setup，不加入一次性性能脚本到上游 tests。无公共 API/用户行为变化，因此无需新增使用文档。

## 4. PR Contents and Publication

按上游模板准备，声明只涉及 mask 存储；提供配置、原始数据、脚本和复现命令，计时列出样本数与波动。不把 mask storage 写成整模型 VRAM，也不把局部构建耗时写成整模型吞吐。不关闭整个 #62350。

认证 REST 读取 issue 与全部 comments，开放 PR 搜索 `62350 in:body`、`audio mask in:title`、`AudioFlamingo3 in:title` 均无重复实现；gh 2.4.0 的 retired Projects classic 字段错误已说明。无现有 PR，因此没有待关闭历史 review threads。

用户确认“已逐行审阅并理解，可以发布 PR”，此前相关测试由用户执行并报告通过。PR 明确 AI 辅助和人类责任，提交使用 DCO Signed-off-by 与 AI attribution trailer。将允许维护者编辑；CI/维护者审阅状态发布后单独核查，不写作本地已通过。

原始验证数据放在用户 fork 的独立 validation 分支，不加入上游三文件补丁。实验审计见 `experiment-audit.md`，公开复现说明见 `REPRODUCE.md`。
