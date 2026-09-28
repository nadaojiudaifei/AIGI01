# V2 实际验证报告

日期：2026-09-28。源码、接口测试与真实 GPU 运行分别记录，不互相冒充。

## 1. GitHub 状态

本轮实际写入成功；原版 DiT-IC 已先导入到 AIGI01，原始文件导入提交为
`4a9c5c2018d3b007163b75aef793aef6a5666851`，上游 pin 为 `cd43f5d9761fb34f5224622145629d3ff2b89ca1`。
源码发布工作流在解包增量前后核对原文件 SHA256，并且禁止增量覆盖原文件与 UPSTREAM_LOCK。
最终增量提交和源文件校验结果以仓库提交及 Actions artifact 内的 COMMIT.txt/source_validation.json 为准。

## 2. 本地实际执行

本轮 CPU 回归测试已经运行，结果见 cpu_tests.txt/cpu_tests.xml；详细环境与命令写入 validation.json。
检查包括原来的数学/因果接口/容器/配置，以及新增的确定性视图、AdamW/LR/GAN 系数、
QA 与压缩清单隔离、OCR 完整参考集合、官方评分命令、盲测去标识化与聚类统计、计数钩子和恢复状态比较。
模块 FLOPs 的单元测试实际调用当前 PyTorch 对一个小型 Linear 层计数，不是预填报告。
新增脚本与配置通过源码级检查；CLI 所有子命令的帮助页都被实际调用。

使用了现有 Python/PyTorch，没有创建 Conda、安装包、下载权重/数据，也没有远程 GPU 凭据。
只读 doctor 的失败原因保留在 local_preflight.json，不会为了“全绿”隐去缺失 GPU 和目标依赖。

## 3. 哪些测试不是生产验收

test_causal_loop.py 的概率表和微型网络是测试替身，不是真实 CompressAI rANS/SANA。
新测试的图片、偏好回答、问答也只是合成夹具，不是科研观测或真实用户数据。
源码发布工作流只做原文件保护、语法、配置和打包检查，不安装训练环境，也不把这些称作 GPU 通过。

## 4. 本轮补齐的源码能力

OCRBench V2：HF/本地准备、原图和重建图的本地固定 VLM、断点预测、固定官方 evaluator commit、
逐题官方评分及原始总分日志；问题/答案不进入压缩提示词。
效率：无插桩计时与独立分模块诊断、注册算子 FLOPs、未计数算子清单、NFE 和显存。
人工研究：匿名盲测页面、完整面板输入校验、参与者 bootstrap、符号翻转检验。
训练：AdamW、0.5 阶段衰减、自适应 GAN、全 microbatch 判别器积累、固定随机视图、清单内容指纹、精确恢复验收。

这些都有实现与对应测试或静态检查，但不等于在大型真实模型上已经执行通过。

## 5. 尚未执行的真实环境验证

SANA/Gemma/DiT-IC/Qwen 权重加载、真实数据完整导出、教师缓存、GPU 前反向、
rANS 独立进程重建、长训练恢复、GAN、DDP、完整图像指标权重、OCR 官方依赖和全数据任务、
4096 分辨率显存/效率、人工参与者收集及最终率失真结果均未在当前会话完成。

远程 gate：

```bash
python -m aigi acceptance --config aigi_configs/local_joint.json \
  --output aigi_runs/acceptance --extended --gpus 2 --execute
```

按实际 GPU 数设置；单卡用 1。有限 smoke gate 不证明收敛，且小样本不会生成可靠 FID/KID。
完整矩阵另跑 `matrix` / `suite` / `run`。只有实际日志证实成功后才能将相应能力标为生产验证通过。

## 6. 明确范围

cadc_style 不是完整 CADC 原模型。作者未公开的精选数据选择和未明确的 OCR VLM 细节不会被编造。
FLOPs 已有模块统计，但不声称支持全部浮点操作。新训练协议有明确默认值和可复现增强，却不是作者每一项细节的完全复制。
没有已训练的新方法权重、论文实测改进、真实人工评分或“任何 GPU 上全部功能必定正常”的保证。

上一版 403/70 项测试记录单独保存在 history/，仅供历史审计，不代表当前 GitHub 状态。
