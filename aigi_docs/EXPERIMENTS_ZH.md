# 实验、完整组件开关与原 DiT-IC 对照

## 1. 三种不同的对照，不要混淆

**作者结果**：原论文和作者公布的数字。本工程不把它们当作已在当前服务器重测的结果。

**原模型推理对照**：`method=ditic`、不提供 AIGI01 `--checkpoint`、使用作者 merged qN 权重，复用原编码器、熵模型与解码公式。因为本工程加入统一文件头与 CPU FP32 熵路径，总 bpp、计时口径仍与原脚本不同。对照时同时报告 payload bpp 和总 bpp。

**统一训练协议对照**：所有方法使用本工程训练器、相同数据划分、同一参数预算规则、相同质量点、相同指标程序。它适合隔离新增模块的影响，但不是逐字逐项复现作者训练脚本。

原项目文件完全保留，因此作者原入口仍可用于原生复核。示例：

```bash
CUDA_VISIBLE_DEVICES=0 python compress.py \
  --config_path configs/inference_merge.yaml \
  --codec_path /data/models/ditic/q3_merge_ema.pt \
  --img_path /data/Kodak --rec_path aigi_runs/original_kodak/rec \
  --bin_path aigi_runs/original_kodak/bin --use_merge --save_img
```

这是原项目命令，不自动获得新增训练器的总字节统计、独立 .aigi 格式或安全参数检查；执行之前阅读保留的原 `ReadMe.md`。本次未在 GPU 上执行该命令。

## 2. 已提供和未提供的原论文实验类别

| 类别 | 当前工程支持情况 | 需要注意 |
|---|---|---|
| Kodak、DIV2K validation、CLIC2020 test 原分辨率重建 | 已有统一清单、真实编码/解码入口 | Kodak 有登记 HF 镜像；后两者用正确官方本地测试集，不能拿 DIV2K train800 代替 val100 |
| PSNR、MS-SSIM、LPIPS Alex、DISTS | 已有评测实现 | 输入尺寸不静默缩放；真实权重尚待远程缓存与验收 |
| NIQE、CLIPIQA、MUSIQ | 已有评测实现 | 预训练指标权重需要 warmup，错误不替换成零 |
| 原 patch-FID / KID | 调用未修改的原 patch 提取 helper | 少于50张不报告 FID；KID还需至少1000 patch；这比原脚本增加了小样本门槛 |
| 四质量点 RD / BD-rate | 已实现 | 使用共同原图集合与相同指标环境，只在质量交集计算，不外推 |
| self-distillation、latent prompt、variance-flow 消融 | 已有各自适用路径开关 | variance-flow 在 joint 时间场路径中不使用，因此不做无效 joint 开关实验 |
| LoRA rank、full-finetune、随机 DiT、去 GAN、去 DISTS | 已有矩阵变体 | 优化器/裁剪/训练过程与作者仍有差别 |
| 60K / 256 分辨率消融预算 | `--protocol ablation256` 已实现 | 只对齐分辨率/步数预算，不伪称其余协议全部相同 |
| 1024/2048/4096 端到端复杂度 | profile 入口已实现 | 无插桩端到端计时、逐模块诊断计时、参数、NFE、显存及注册算子 FLOPs；未计数算子清单同时输出 |
| 人工偏好实验 | 已有盲测材料与结果汇总 | 支持参与者聚类 bootstrap 与符号翻转检验；必须提供真实回答，不生成伪造分数 |
| OCRBench V2 文字理解任务 | 已有图像压缩→固定 VLM→官方评分链路 | 保留原题型/字段，完整步骤见 V2_USAGE_ZH.md；不是普通 OCR 替代 |
| 自然/AI图像与多生成器扩展 | 数据清单与通用实验路径已提供 | 本次未训练，尚无泛化结论 |

因此不能把这一版称为“DiT-IC 所有实验已经完整复现”。表中待实现的功能和远程待验证的功能是两种不同状态，不能相互掩盖。

## 3. 全部组件开关的实际作用域

通过 JSON 的 `components` 设置，或者使用 `configure --set components.NAME=false`。未知开关会报错，不静默忽略。

| 开关 | 关闭后做什么 | 主要适用方法 |
|---|---|---|
| learned_token_weights | 有效 token 使用等权，不使用学习分数 | idea1/idea2/joint 的地图聚合 |
| map_distillation | 不加教师地图目标；仍加载缓存中的文本特征 | idea1/idea2/joint |
| texture | m_texture 固定为1 | cadc_style/idea1/joint |
| importance | 去掉量化控制的 w、S 输入，以及分配目标中的 w 与 S 蒸馏项 | idea1/joint；不是关闭所有解码语义功能 |
| difficulty | c 不输入修正分支/分配目标，不计算 c 蒸馏 | idea1/joint |
| explanation | e 不输入修正分支，不计算 e 蒸馏 | idea1/joint |
| correction | δ 固定为0 | idea1/joint |
| text_entropy | 不向熵上下文添加文本 cross-attention | idea1/joint；真实文本仍可用于地图与生成 |
| waterfill | 不加局部比特分配损失 | idea1/joint |
| weighted_loss | 图像 MSE/LPIPS 使用均匀权重 | idea1/joint |
| snr_time | 空间 SNR 换成该图全局平均 SNR | idea2/joint |
| semantic_time | 时间方程中的 beta 设为0 | idea2/joint；不删除双路引导里的语义权重 |
| time_residual | rho 固定为0 | idea2/joint |
| local_adaln | 不安装局部 norm 残差仿射 hooks | idea2/joint |
| dual_guidance | 推理只算 vpy，DiT NFE从3变1 | idea2/joint；训练本来只算1分支，dropout目标不变 |
| self_distillation | 去掉 latent cosine hinge 对齐损失 | 所有 codec 路径 |
| latent_prompt | 不附加压缩潜变量 prompt；原基线用零条件替换 | 所有路径 |
| variance_flow | 改用固定原始 sigma 的更新，不用预测标准差调节 | ditic/cadc_style/idea1，不适用于idea2/joint |

“关闭 importance”不是把所有 w 从整个项目删除。weighted_loss、semantic_time 和 dual_guidance 是独立职责。需要研究“全部语义信息关闭”时，应明确组合相关开关，并使用新的实验名称，而不是给一个开关赋予它代码中没有的含义。

## 4. 单独组件实验与消融训练

单个组件不能凭空独立压缩 RGB；地图、熵条件、时间场依赖共同骨干。工程提供的是在固定骨干与必要前置组件上单独训练/测试其影响，不声称每个卷积块本身是一台完整压缩器。

地图预测器可以独立 `stage=maps` 训练。其余组件用 leave-one-out（完整模型去一项）及 add-one（最小该家族框架中增加一项）矩阵。

```bash
python -m aigi configure --base aigi_configs/local_joint.json \
  --output aigi_configs/no_text_entropy.json \
  --set 'experiment=joint_without_text_entropy' \
  --set 'components.text_entropy=false' \
  --set 'output=aigi_runs/joint_without_text_entropy_q3' \
  --set 'maps_checkpoint=""'
python -m aigi train --config aigi_configs/no_text_entropy.json
```

禁止直接将完整模型 checkpoint 配上不同结构开关后冒充“该消融单独训练完成”。正式消融应分别训练，并记录初始化来源。推理会检查主要 checkpoint/config 字段一致性。

## 5. 自动矩阵

五条主路径，四质量点：

```bash
python -m aigi matrix --config aigi_configs/local_joint.json \
  --output aigi_runs/main_comparison --qualities 1 2 3 4 \
  --no-ablations --no-add-one --gpus 1 \
  --datasets aigi_data/kodak/manifest.jsonl aigi_data/div2k_val.jsonl aigi_data/clic2020_test.jsonl
```

默认主协议：stage1=100K步、256中心裁剪、无GAN；stage2=60K步、512中心裁剪、GAN基础权重0.1（默认梯度比自适应），后者从本变体自己的 stage1 权重初始化。GAN从本阶段30%进度后启用。全局有效 batch 由你的配置与 GPU 数决定，不自动改成作者使用的 batch。

含 leave-one-out、add-one、共同基线组件与 rank 变体：

```bash
python -m aigi matrix --config aigi_configs/local_joint.json \
  --output aigi_runs/ablations_q3 --qualities 3 \
  --protocol ablation256 --max-jobs 1000
```

`ablation256` 是60K步、256 crop、GAN基础权重0.1，默认 gan_mode=adaptive；`without_gan`变体为0。optimizer=adamw、lr_gamma=0.5，其他作者协议差异仍须报告。

```bash
python -m aigi run --plan aigi_runs/ablations_q3/matrix.json
python -m aigi run --plan aigi_runs/ablations_q3/matrix.json --execute
```

计划生成不开始训练，运行器不创建环境，也不自动下载缺失模型和数据。它记录 config、argv、依赖文件、退出码、耗时和日志。配置或依赖 checkpoint 改变后，已完成任务不会仅凭同名文件就被当成相同实验。

### Add-one 的前置依赖

若“只开 difficulty”却关闭 correction 和 waterfill，那么 c 不可能影响编码；这种实验没有意义。因此 `only_difficulty` 同时启用 correction，`only_explanation` 同时启用 correction；`only_learned_token_weights` 启用 importance；`only_waterfill` 启用 importance+difficulty；`only_weighted_loss` 启用 importance；`only_local_adaln` 启用 snr_time；`only_map_distillation` 启用 importance+difficulty+explanation+correction。

所以 `only_X` 的准确意思是“X 与它所需的最小前置项”，不是掩盖其他开关的绝对单模块。生成的 JSON 会完整写出这些布尔值。

### 全因子组合

```bash
# 思路二：5个家族开关，32种组合，另含5条主路径。
python -m aigi matrix --config aigi_configs/local_joint.json \
  --output aigi_runs/factorial_idea2 --qualities 3 --protocol ablation256 \
  --no-ablations --no-add-one --factorial idea2 --max-jobs 1000

# 思路一：10个家族开关，1024种组合；预算必须显式扩大。
python -m aigi matrix --config aigi_configs/local_joint.json \
  --output aigi_runs/factorial_idea1 --qualities 3 --protocol ablation256 \
  --no-ablations --no-add-one --factorial idea1 --max-jobs 2000
```

完整布尔笛卡尔积包含依赖项关闭后等效的组合，例如 correction 关闭时 explanation 对量化分支不起作用。**1024个配置不等于1024个独立创新机制。** 分析时应按生成的实际开关归类，并优先解释前置依赖闭合的 add-one 和 leave-one-out。

所有质量点的默认 `rate_weight` 为 q1=16、q2=2、q3=0.5、q4=0.25。这是本工程明确的质量扫描表，不能把它声称为作者所有 checkpoint 的精确训练超参数对应表。

## 6. 正式数据与结果口径

自然训练集和AI训练集分别建立清单，先去重再划分。对 DiffusionDB 的相同原始 prompt 进行组划分；14M包含2M，不能将两者不去重直接跨 train/test 放置。多生成器数据建议按 generator 分开写清单，做训练生成器与未见生成器测试；本版不自动替你决定生成器留出政策。

所有方法使用同一组原图 ID / SHA256。对自然图像生成的 BLIP prompt 使用同一份固定清单；基线不消费 prompt，新方法的 prompt 传输计入总码率。测试图保持原尺寸，只做 padding/crop-back；不能对某方法额外降采样。

样本总 bpp 是总 bits/总 pixels，同时保留逐图 bpp 的平均，两者对尺寸不同的数据不相同。输出图片统一 uint8 RGB PNG，指标在保存后的 PNG 上计算，避免内存浮点重建和实际保存图之间口径不一致。

## 7. 率失真与 BD-rate

完成至少四个实测质量点后：

```bash
python -m aigi compare \
  --inputs aigi_runs/kodak_ditic_q1/metrics.json aigi_runs/kodak_ditic_q2/metrics.json \
           aigi_runs/kodak_ditic_q3/metrics.json aigi_runs/kodak_ditic_q4/metrics.json \
           aigi_runs/kodak_joint_q1/metrics.json aigi_runs/kodak_joint_q2/metrics.json \
           aigi_runs/kodak_joint_q3/metrics.json aigi_runs/kodak_joint_q4/metrics.json \
  --baseline ditic --metric psnr --output aigi_runs/bd_psnr.json --plot
```

这里的路径示例应对应你实际的推理结果目录；矩阵的输出目录可从 `matrix.json` 的 expected 字段直接读取。不要照抄不存在的路径后认为脚本会自动定位。

代码在 log(rate)–quality 曲线上做 PCHIP 积分，只取两条曲线的质量交集。要求至少4点、质量严格变化、码率同方向单调、有限值、同原图集合、同指标版本；不满足时拒绝产生一个看似精确的 BD-rate。负数表示在质量交集内新方法需要更少码率。LPIPS/DISTS 越低越好时内部转换方向，不直接套 PSNR 的方向。

## 8. 效率与人工研究

```bash
python -m aigi profile --config aigi_configs/local_joint_stage2.json \
  --checkpoint aigi_runs/joint_q3_stage2/last.pt --ema \
  --output aigi_runs/joint_profile.json --sizes 1024 2048 4096 --repeats 5
```

profile 使用确定性合成测试图，记录端到端墙钟时间；正式效率论文表还应在真实图像上补充测试。计时包括 CPU 熵编解码和搬运，不包括模型加载与文本编码启动。双路引导必须报告 NFE=3，不能只写“单步”让读者以为仅一次网络计算。

```bash
python -m aigi study --a aigi_runs/kodak_ditic_q3/inference.json \
  --b aigi_runs/kodak_joint_q3/inference.json --output aigi_runs/blind_study
```

盲测材料随机左右顺序并隐藏方法名；配对总码率差需在预设容差内。不要将答案映射文件发给参与者，不要声称生成页面等于已完成用户研究。V2 已增加 study-results：完整参与者×图像面板、参与者聚类 bootstrap 置信区间、参与者层级双侧符号翻转检验和图像聚类敏感性区间。样本数、参与人数、显示条件和统计方案仍须在真正实验前固定。

## 9. 可复现性材料

每组实验至少保存：`UPSTREAM_LOCK.json`、`assets.lock.json`、模型/文本/代码 fingerprint、完整 resolved_config、数据清单和划分报告、teacher cache contract、pip freeze、训练日志、原始 .aigi、重建 PNG、逐图 metrics 与矩阵状态。

```bash
python -m pip freeze > aigi_runs/pip_freeze.txt
python aigi_scripts/bootstrap_upstream.py --destination . --verify-only
```

把 checkpoint、数据和缓存放在 Git 之外，避免把大文件或私有图像意外提交。任何性能提升与泛化结论都应来自这套记录下的实际实验，而不是本文设计说明。
## 10. V2 补充实验计划

`python -m aigi suite --matrix BASE_MATRIX --output OUTPUT` 为每个变体的末阶段生成效率任务。
同时提供 `--ocr-questions --ocr-model --ocr-official --ocr-manifest` 会增加原图 VLM 评测，
以及每个方法/质量点的真实压缩、重建图 VLM 评测和官方 OCRBench V2 评分。
这些任务使用同一个 VLM、输入预处理和生成参数；QA 文本不会成为压缩器的提示词。
生成计划不是执行，仍通过 `python -m aigi run --plan OUTPUT/matrix.json --execute` 运行。

默认 center 裁剪可换成 cached_random 固定视图库，但必须为所有方法使用相同视图设置并重建对应教师缓存。
运行器除任务配置/依赖外，还核对源码、数据清单和输出文件哈希；已完成任务的同名输出变动后不会被静默当作旧成功结果。

“实验类别已有入口”和“论文完整数值已经复现”是两回事。本版仍没有真实 GPU 结果。
FLOPs 明确是注册算子口径，不能把其中未计数的算子按零成本解读。
