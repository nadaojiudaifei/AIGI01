# V2：把新增实验真正串起来

本文件补充 [从零运行指南](QUICKSTART_ZH.md)，不替代 [算法说明](ALGORITHM_ZH.md)。
所有命令在远程 GPU Linux 的 AIGI01 仓库根目录执行；本文没有在聊天环境安装 Conda、训练依赖、模型或数据。
源码检查、CPU 测试、真实 GPU 验收、完整科研实验是四种不同层次，不能互相冒充。

## 1. 已有仓库，直接克隆

```bash
git clone https://github.com/nadaojiudaifei/AIGI01.git
cd AIGI01
export PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python aigi_scripts/bootstrap_upstream.py --destination . --verify-only
python aigi_scripts/validate_source.py --root . --require-upstream \
  --output aigi_runs/source_validation.json
```

原仓库把部分 `.pyc` 也纳入版本控制，所以必须避免重新写入原目录的 bytecode。`aigi` 包入口也会设置 `sys.dont_write_bytecode=True`。
源码校验器只使用标准库和 Bash，不安装软件，不运行训练。不要再对已经有内容的 AIGI01 使用早期 bootstrap 的 `--push`。

环境安装按照 QUICKSTART 第 2 节进行。安装完先运行 `pip check`、`python -m aigi doctor --require-gpu` 和 CPU 测试。
`aigi_requirements.txt` 是明确目标版本集合，不是假装已经在你的 GPU 型号上求解和实测过的锁文件。

## 2. 先准备小数据和模型

```bash
python -m aigi models --output aigi_assets --qualities 3
python -m aigi models --output aigi_assets --qualities 3 --download

python -m aigi data-hf div2k_train --output aigi_data/div2k_train
python -m aigi data-hf diffusiondb_1k --output aigi_data/diffusiondb_1k
python -m aigi data-hf kodak --output aigi_data/kodak
```

Kodak 只用于测试，不混入训练。数据导出、本地文件夹清单、caption、去重和 train/val/test 组划分，详见原运行指南。
自然图像补充 BLIP caption；已有原始生成提示词的 AI 图像保留原提示词。不要把描述器生成的句子伪称为真实生成 prompt。
离线机器用 `--local-source` 或 `data-local`；SANA、ELIC、DiT-IC 路径通过 configure 指向现有本地文件，不必下载。

完成原指南中的清单后生成配置：

```bash
python -m aigi configure --base aigi_configs/joint_q3.json \
  --assets aigi_assets --output aigi_configs/local_joint.json \
  --set 'train_manifest=aigi_data/splits/train.jsonl' \
  --set 'val_manifest=aigi_data/splits/val.jsonl' \
  --set 'teacher_cache=aigi_cache/teacher' \
  --set 'output=aigi_runs/joint_q3'
```

## 3. 确定性多视图增强

中心裁剪依旧可用。需要增强时：

```bash
python -m aigi configure --base aigi_configs/local_joint.json \
  --output aigi_configs/local_joint_views.json \
  --set 'augmentation=cached_random' --set 'views_per_image=4'
python -m aigi teacher --config aigi_configs/local_joint_views.json \
  --manifest aigi_data/splits/train.jsonl
python -m aigi teacher --config aigi_configs/local_joint_views.json \
  --manifest aigi_data/splits/val.jsonl
```

每张训练图有 4 个固定随机视图，包含随机裁剪与水平/垂直翻转。它们由图像哈希、crop、视图编号、seed 确定。
教师处理实际变换后的图像，缓存键包含完整视图参数。训练不会误用另一张裁剪图的教师标签。
配置中 `val_manifest` 对应的教师及验证始终是中心裁剪。更改 crop/seed/views 后重新生成缺少的缓存即可。
这是有限、可复现视图库，不等于作者每个 epoch 无限重新采样的增强策略；所有比较方法必须共享视图规则。

## 4. 优化器、GAN 和断点恢复

V2 默认：AdamW，betas=(0.9,0.999)，weight_decay=0.01；
初始学习率 1e-4，在阶段完成 50%、80%、90% 时分别乘 0.5。辅助量化分位点优化器是 Adam，学习率 1e-3。
GAN 默认按重建梯度/对抗梯度的范数比自适应加权；`gan_weight` 是基础系数，`gan_mode=fixed` 才使用固定权重。
判别器使用全部累积 microbatch 的平均梯度，不是只用最后一个 microbatch。

```bash
python -m aigi train --config aigi_configs/local_joint.json
python -m aigi train --config aigi_configs/local_joint.json \
  --resume aigi_runs/joint_q3/last.pt
```

用于验收中断恢复，而非改变训练预算：

```bash
python -m aigi train --config aigi_configs/local_joint.json --stop-after 10
python -m aigi train --config aigi_configs/local_joint.json \
  --resume aigi_runs/joint_q3/last.pt
```

`stop-after` 是绝对优化步。它不会改 `steps`，因而不会改变学习率时间轴。
恢复必须保持世界大小、算法、优化器配置、数据清单内容等一致。旧 V1 checkpoint 缺少数据 SHA 时会拒绝“精确恢复”声明。
所有模型状态都存在，但已完成训练的 checkpoint 不应该再次当作未完成恢复点；新阶段使用 `init_checkpoint`。

## 5. 真实服务器验收

```bash
python -m aigi acceptance --config aigi_configs/local_joint.json \
  --output aigi_runs/acceptance --extended --gpus 1
python -m aigi acceptance --config aigi_configs/local_joint.json \
  --output aigi_runs/acceptance --extended --gpus 1 --execute
```

第一条只生成计划。第二条使用真实依赖与真实权重，依次执行五条方法的两步训练、真实熵编码、两个新进程的独立解码，
随后比较完整两步训练与第一步中断后恢复的模型、优化器、EMA、RNG 和数据哈希。
`--extended` 增加 GAN 梯度累积及完整指标入口；`--gpus 2` 另加两卡 DDP 冒烟测试。需要至少 max(2,gpus) 张训练图和一张不重叠验证图。

判别器短训练通过不等于 GAN 收敛。单张验证图不足以计算可靠 FID/KID，所以这两项会有不可用原因，不填零。
模型下载、完整大数据跑分、长训练和 4096 分辨率显存仍需单独验收。
某个任务失败后保留日志并立即中止；只有全部请求的任务成功才写 `acceptance.json` 的 `passed=true`。

## 6. OCRBench V2：数据准备

官方本地数据路径通常含 `EN_part/`、`CN_part/`、`OCRBench_v2.json`：

```bash
python -m aigi ocr-prepare \
  --annotations /data/OCRBench_v2/OCRBench_v2.json \
  --images-root /data/OCRBench_v2 --output aigi_data/ocr_v2/prepared
```

也可以从作者的 HF 镜像导出：

```bash
python -m aigi ocr-data --output aigi_data/ocr_v2
# 已有 parquet/save_to_disk 目录时：
python -m aigi ocr-data --output aigi_data/ocr_v2 \
  --local-source /data/downloaded_ocrbench_v2
```

固定数据提交：`c7e7cdf23bdb6774661e9b0caf0d9935a42feb8b`。使用 `--limit` 的结果只能标成子集，不是完整 10K 评测。

输出分工：
- `prepared/manifest.jsonl`：去重后的图像压缩清单，prompt 默认为空。
- `prepared/questions.json`：完整 QA、题型、答案和图像对应关系。
- `prepared/prepare.json`：样本数、文件哈希、协议说明。

压缩器绝对不读取题目或答案。空提示词方案是一个明确的测量口径；需要图像 caption 时，只能从图像生成并对全部方法固定使用同一份清单。

## 7. OCRBench V2：统一评测模型和官方评分

```bash
python -m aigi ocr-assets --output aigi_assets/ocr
python -m aigi ocr-assets --output aigi_assets/ocr --download
```

下载的评测器是 `Qwen/Qwen2.5-VL-3B-Instruct`，实际 HF commit 与内容指纹写入资产锁；
官方代码固定在 `0ea56c2503a2d940700d581b777f23c16de0eafd`。
VLM 是本工程确定的统一选择，不能写成原 DiT-IC 作者的精确评测器复现。预测只从本地模型加载，不自动联网。

先评原图：

```bash
python -m aigi ocr-predict \
  --questions aigi_data/ocr_v2/prepared/questions.json \
  --model aigi_assets/ocr/vlm --output aigi_runs/ocr_original/predictions.json
```

再对每个已训练方法做实际压缩，随后在重建图上回答同一批题：

```bash
python -m aigi infer --config aigi_configs/local_joint.json \
  --checkpoint aigi_runs/joint_q3/last.pt \
  --manifest aigi_data/ocr_v2/prepared/manifest.jsonl \
  --output aigi_runs/ocr_joint/images

python -m aigi ocr-predict \
  --questions aigi_data/ocr_v2/prepared/questions.json \
  --model aigi_assets/ocr/vlm \
  --inference aigi_runs/ocr_joint/images/inference.json \
  --output aigi_runs/ocr_joint/predictions.json
```

所有方法固定：greedy 解码，max_new_tokens=1024，min_pixels=200704，max_pixels=1003520，seed=903。
这是评测 VLM 的统一图像预处理预算，不是压缩器擅自改变原图尺寸。修改预算必须对所有方法一起修改。
预测支持按 QA 断点恢复，检查题目、图像内容、VLM、参数和运行环境指纹；失败题不会被静默删掉。

官方评分脚本有其独立依赖，需要在远程按官方 `OCRBench_v2/requirements.txt` 手动配置。
`--eval-python` 可指定你已经准备好的评测解释器，避免强行污染压缩训练环境。本工程不自动创建第二个 Conda。

```bash
python -m aigi ocr-score \
  --predictions aigi_runs/ocr_joint/predictions.json \
  --official-root aigi_assets/ocr/MultimodalOCR \
  --output aigi_runs/ocr_joint/official

python -m aigi ocr-score \
  --predictions aigi_runs/ocr_joint/predictions.json \
  --official-root aigi_assets/ocr/MultimodalOCR \
  --eval-python /path/to/existing/ocr_environment/bin/python \
  --output aigi_runs/ocr_joint/official --execute
```

第一条只打印命令。真正运行时会验证官方 Git commit 和评分脚本没有被修改。
保存逐题官方 `scored.json`、两个官方脚本的原始日志与 `report.json`，不会拿 PSNR、普通 OCR 文本准确率或零分替代官方多任务指标。

## 8. 分模块效率

```bash
python -m aigi profile --config aigi_configs/local_joint.json \
  --checkpoint aigi_runs/joint_q3/last.pt \
  --sizes 1024 2048 4096 --repeats 5 --output aigi_runs/joint_profile.json
```

一次测试分为两部分：无插桩的暖机及重复墙钟计时；单独插桩的模块/FLOPs 诊断。
前者才用于方法效率对比。后者记录 VAE 编解码、ELIC、分析/超先验/上下文/合成、地图、文本条件与 DiT 等模块的调用数、
inclusive/exclusive 时间和注册算子 FLOPs。嵌套 inclusive 行不能相加，CPU 熵编解码与传输开销也不能删去。

同时报告未注册浮点张量算子清单，`flops_complete=false`；
矩阵 multiply-add 按两次 FLOP 计，rANS 整数工作不冒充浮点 FLOPs。不能将缺失算子解释成零计算成本。
NFE 真实记录生成调用次数：默认思路二双路引导推理是三次 DiT，不是一次。
首个文本编码调用计时另列；这是缓存状态相关的诊断，不包含大模型加载启动。

## 9. 人工偏好研究与结果汇总

```bash
python -m aigi study \
  --a aigi_runs/kodak_ditic_q3/inference.json \
  --b aigi_runs/kodak_joint_q3/inference.json \
  --criterion realism --tolerance 0.05 --output aigi_runs/blind_study
```

先匹配总码率，超过预定容差会拒绝配对。默认比较视觉真实感，不给原图；`fidelity` 才显示参考图，是不同研究问题。
仅分发 `blind_study/public/`，不要分发 `PRIVATE_KEY.json`。
页面随机左右顺序，去掉图像元数据，匿名参与者在浏览器本地导出回答 CSV；无外部上传。
研究组织者需自行做好自愿参与、显示条件、样本量和统计方案，源码不会虚构参与者。

```bash
python -m aigi study-results \
  --key aigi_runs/blind_study/PRIVATE_KEY.json \
  --answers /data/responses/answers_P001.csv /data/responses/answers_P002.csv \
  --draws 10000 --output aigi_runs/blind_study/results.json
```

要求每位参与者完成全部题目。重复或缺失回答直接报错，不偷偷删样本。
Tie 记半票。主置信区间按参与者聚类 bootstrap，另有按图像聚类的敏感性区间；
双侧符号翻转检验在参与者层级进行，不将同一人多次回答当独立投票。
主区间条件于当前图像集合，图像敏感性区间不是两维联合置信区间。少于两名参与者不报告显著性或参与者区间。

## 10. 一次生成补充实验矩阵

```bash
python -m aigi suite --matrix aigi_runs/main_comparison/matrix.json \
  --output aigi_runs/supplementary \
  --ocr-questions aigi_data/ocr_v2/prepared/questions.json \
  --ocr-manifest aigi_data/ocr_v2/prepared/manifest.jsonl \
  --ocr-model aigi_assets/ocr/vlm \
  --ocr-official aigi_assets/ocr/MultimodalOCR \
  --eval-python /path/to/existing/ocr_environment/bin/python

python -m aigi run --plan aigi_runs/supplementary/matrix.json
python -m aigi run --plan aigi_runs/supplementary/matrix.json --execute
```

先完成主矩阵训练。补充矩阵为每个变体/质量点的末阶段增加效率、OCR 压缩、VLM 问答和官方评分。
人工研究不能自动跑出真实回答，必须另行收集后才汇总。
运行器校验源码、配置、输入清单、依赖 checkpoint 和最终输出哈希；失败立即停止，不把已有同名文件当成功。

## 11. 仍然需要真实验证的部分

本工程没有在当前会话执行真实 SANA/Gemma/Qwen 权重、CompressAI rANS、CUDA 训练、DDP、GAN、全套感知指标或大型数据导出。
完整 CADC 不是 cadc_style；作者 150K 精选清单不能从未公开的选择过程自动恢复；有限视图库也不是无限随机增强。
这些不是“二选一待定设计”，而是已经明确的实现范围和实验事实边界。
