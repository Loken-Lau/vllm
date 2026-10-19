# 实验审计：audio encoder compact additive mask

结论：本次声明的 mask 存储优化与输出等价已有实测支持，未发现阻止发布的代码或实验问题。不宣称整模型显存减少 1500 倍，也不宣称端到端加速。上游 CI 与维护者审阅尚未完成。

## 证据清单

| 要验证什么 | 实际实验与结果 | 证据 |
| --- | --- | --- |
| 回归用例能抓到原问题 | 原始两份源码 + 新测试，4 项按预期因平方 mask 失败；修改后 7 passed | `pytest-audio-masks-baseline.log`, `pytest-audio-masks-latest.log` |
| 最低依赖兼容 | Transformers 5.16.1 单独安装目标目录，7 passed；主环境 5.19.0 保持不变 | `pytest-audio-masks-tf516.log` |
| 真实 encoder 语义 | 两个 Transformers 版本各 96 项，合计 192 项；两模型，CPU FP32 / CUDA FP32、FP16、BF16，默认/SDPA/eager，单条及混合奇偶有效长度；最大输出差异 0 | `audit-audio-mask-matrix{,-tf516}.json` |
| 边界和测试有效性 | 每个版本六组 mask-builder 边界；负对照改变 padding values 不影响 masked 输出，去掉 mask 则明显改变输出 | 同上 JSON 的 `boundary_cases`, `padding_control` |
| 实际存储收益 | A800 BF16 / S=1500，B=1/4/16，实际 tensor storage 4.5/18/72 MB → 3/12/48 KB；记录分配峰值；SDPA 输出差异 0 | `benchmark-audio-masks.json` |
| 局部构建耗时 | AF3 builder 每组 5 次独立 blocked_autorange，保留中位数/IQR；B=1 没有加速 | 同上 JSON |
| 预训练完整模型路径 | 两模型、单请求/混合 batch、独立 baseline/compact 引擎，每种情况预热 1 次 + 测量 10 次；80 对请求的文本、token IDs、top-5 logprobs 完全一致 | `e2e-audio-models-comparison.json` 及四份原始 JSON/log |
| 真实路径确实用了修改 | hooks 确认实际 SDPA encoder；原始 `[B,1,1500,1500]` → 修改 `[B,1,1,1500]`，每块 mask storage 4,500,000 → 3,000 bytes | 同上 JSON 的 `captured_masks` |
| 长音频分块 | AF3 35 秒输入，实测编码总块数比 Qwen 多；包括预热共 55 个 AF3 音频块（Qwen 44），吻合 30 秒分块 | 同上 JSON |
| 性能退化排查 | 同进程原始/修改/修改/原始（ABBA），每轮每负载预热 5 次、测量 20 次；每版本每负载 40 个样本，全部输出及 logprobs 一致 | `*-abba.json`, `abba-comparison.json` |
| 同步、格式、类型、CI接入 | 真实推理启用 VLLM_GPU_SYNC_CHECK=error；适用 pre-commit 全通过，新增测试接入现有 processing CI | 四份 E2E log、`pre-commit-audio-masks-latest.log` |

## 完整模型延迟

独立进程的最初计时有首测毛刺、且两个模型的变化方向相反，因此保留原始值并补做同进程 ABBA，未通过删掉异常值来制造加速。

下表每格为 40 个测量样本的中位数 ± 总体标准差，单位 ms；完整逐次样本与两轮中位数均保留。

| 模型 / 负载 | 原始 | 修改 | 中位数变化 |
| --- | --- | --- | --- |
| Qwen2-Audio / Single | 216.41 ± 2.88 | 214.53 ± 3.40 | -0.87% |
| Qwen2-Audio / Mixed | 562.61 ± 6.08 | 564.11 ± 3.04 | +0.27% |
| AudioFlamingo3 / Single | 224.46 ± 3.23 | 222.55 ± 2.78 | -0.85% |
| AudioFlamingo3 / Mixed | 599.71 ± 20.52 | 602.21 ± 7.30 | +0.42% |

这些样本的范围重叠，变化幅度小，不作为吞吐提升证据，也不是统计非劣效检验。能够直接确认的收益是 mask 的实际分配减少。

## 范围与局限

- 完整模型：A800 / BF16 / TP=1 / V2 runner / eager execution / greedy max_tokens=32。关闭 prefix cache 和 multimodal processor cache。encoder 实际后端为 SDPA。
- 音频来自一份 15.948 秒真实语音；短样本是前 2 秒，29/35 秒长样本重复该语音。不是多个独立录音，也不是数据集级 WER 评测。
- encoder 矩阵是小型随机初始化真实 encoder；与预训练完整模型对照分别报告。
- 零有效长度仅验证 mask-builder，不声称完整模型支持空音频。奇偶长度指有效帧长度；Qwen 实际 processor 使用固定 mel 宽度。
- 未验证其他 GPU 架构、TP、多种量化、encoder compile/CUDA graph、新 FlashAttention encoder 分派。它们不作为本 PR 的新增能力。
- 未运行全仓库测试。运行的是受影响的 processing 文件、补充兼容矩阵与完整模型 A/B；上游 CI 状态另行报告。
- Python 源码基线为 `283d48f4ebb404b00e76264ad8f759d7e9a05dfd`；最初微基准/原始回归基线为 `fb61ac07b`，三份受影响文件在两者之间完全一致。
- 原生扩展使用官方 nightly `852989f580820849113d67d13fd8053e279d0444`，与本地 Python 源码 revision 分别披露。

## 发布前复核

用户已明确确认“已逐行审阅并理解，可以发布 PR”，并报告相关测试通过。本轮没有再修改上游三个文件；SHA-256 与此前人类审阅版本一致。完整复现入口为 `REPRODUCE.md`。所有未执行项都限制了结论范围，没有被填为通过。
