# AIGI01：基于原 DiT-IC 的增量研究工程

目标仓库：`nadaojiudaifei/AIGI01`。原始 DiT-IC 固定为 `cd43f5d9761fb34f5224622145629d3ff2b89ca1`。
本仓库先导入原项目，再增加 `aigi/` 等目录；原文件逐字节保持不变。上游导入提交是 `4a9c5c2018d3b007163b75aef793aef6a5666851`。
`aigi_docs/UPSTREAM_LOCK.json` 保存原文件 SHA256。不要修改原 `models/`、`configs/`、训练脚本或原 `ReadMe.md`。

## 先读什么

| 文档 | 用途 |
|---|---|
| [从零配置到训练、编解码](aigi_docs/QUICKSTART_ZH.md) | 在远程 GPU Linux 上手动配置 Conda、下载数据和模型、训练与推理 |
| [实际算法逐步解释](aigi_docs/ALGORITHM_ZH.md) | 参数初次出现的含义、教师地图、编码、码流、解码和损失 |
| [V2 新增功能与命令](aigi_docs/V2_USAGE_ZH.md) | 确定性增强、精确恢复验收、OCRBench V2、效率统计、人工研究 |
| [数据和模型](aigi_docs/DATA_MODELS_ZH.md) | 本地/HF、大小数据集、自然图像和 AI 图像、提示词来源 |
| [实验与消融](aigi_docs/EXPERIMENTS_ZH.md) | 主方法、独立组件、全因子组合、率失真与论文实验覆盖 |
| [实际验证与边界](aigi_docs/VALIDATION_ZH.md) | 通过了哪些测试，真实 GPU 验证还缺什么 |
| [来源](aigi_docs/SOURCES.md) | 原项目、论文、API、数据集和模型来源 |

## 五条方法路径

`ditic` 复用原 DiT-IC；`cadc_style` 是 CADC 式编码前缩放的迁移对照，而非完整 CADC；
`idea1` 实现语义重要性、去噪困难度、空间量化与文本熵条件；
`idea2` 实现解码侧局部 SNR 时间场、语义时间偏置与双路引导；
`joint` 联合两种设计。

已删除人脸/渲染文字先验融合和用户手工 token 权重控制。所有保留的设计均有唯一的默认算法及显式消融开关。
`c/e` 是有限求积得到的代理量，不是精确互信息或真实逐像素比特数。解码器不访问原图、教师缓存或编码端 `m/S/c/e`。

## 已有入口

完整图像压缩训练、地图训练、GAN、DDP、EMA、精确状态恢复、真实 CompressAI 熵编码与独立解码、
HF/本地数据与模型、九类图像指标、RD/BD-rate、消融矩阵、分模块诊断、OCRBench V2 官方评分适配、
盲测材料和真实回答汇总。默认命令不擅自启动大型实验或下载；下载/实验计划需要明确执行参数。

```bash
git clone https://github.com/nadaojiudaifei/AIGI01.git
cd AIGI01
python aigi_scripts/bootstrap_upstream.py --destination . --verify-only
python aigi_scripts/validate_source.py --root . --output aigi_runs/source_validation.json
```

以上校验不安装软件、不创建环境。Conda、模型和数据由你在远程服务器按指南配置。不要向本仓库提交权重、训练图像或私有凭据。

## 不能混淆的完成状态

源码入口和 CPU 回归测试不等于真实 GPU 全链路验收。当前交付没有训练好的新方法权重、论文实测数字或真实参与者结果。
完整 CADC、作者未发布的 150K 精选训练清单、未明确公开的 OCR 评测模型细节不在本实现中被冒充复现。
FLOPs 是 PyTorch 已注册算子的计算估计，输出同时列出未计数算子，不伪称覆盖全部操作。

在真实服务器上运行：

```bash
python -m aigi acceptance --config aigi_configs/local_joint.json \
  --output aigi_runs/acceptance --extended --gpus 1
python -m aigi acceptance --config aigi_configs/local_joint.json \
  --output aigi_runs/acceptance --extended --gpus 1 --execute
```

先按运行指南生成 `local_joint.json`。只有实际运行全部成功才会写 `passed=true`。
两卡服务器改成 `--gpus 2` 可增加 DDP 冒烟测试；这仍不等于长期收敛或论文质量优势证明。
