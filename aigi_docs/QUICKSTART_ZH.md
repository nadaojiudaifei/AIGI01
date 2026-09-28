# V2 使用入口更新

AIGI01 已有原项目与增量代码；正常使用先 `git clone https://github.com/nadaojiudaifei/AIGI01.git`，然后进入仓库。
下面早期 bootstrap 命令只用于**另一个全新空仓库**，不能对已初始化的 AIGI01 再执行 `--push`。
先读 [V2 新增步骤](V2_USAGE_ZH.md)。环境安装仍由你在远程 GPU Linux 上手动执行；本文代码不会在聊天环境创建 Conda。

# 从零开始：远程 Linux GPU 服务器运行指南

本文对应当前包内 CLI，而不是尚未实现的伪命令。先读根目录 `AIGI_README.md` 的状态说明。所有下面的安装、下载、GPU 命令均由你在远程服务器执行；本次会话没有替你创建环境或运行这些 GPU 步骤。

## 0. 准备目录和 GPU

使用服务器管理员提供的 SSH 方式登录 Linux，进入有足够磁盘空间的位置。先检查：

```bash
uname -a
nvidia-smi
df -h
free -h
which git
```

`nvidia-smi` 必须能看到 NVIDIA GPU。驱动需要支持所装的 PyTorch CUDA 构建；`nvidia-smi` 上显示的 CUDA 版本是驱动兼容上限，不代表你安装了对应 CUDA Toolkit。本配置选择 PyTorch 2.8.0 / torchvision 0.23.0 的 CUDA 12.8 轮子。驱动不兼容时请先与管理员处理，不要混装另一套 CUDA 依赖然后仍声称遵循了相同实验环境。

使用 batch_size=1 起步，先做远程 smoke acceptance，再按实测显存调整。Gemma 文本编码器放在 CPU，SANA 与训练图主要使用 GPU；不能仅根据 DiT 参数量推断整套训练的显存和内存需求。大规模教师缓存还会显著占用磁盘。

## 1. 导入并上传原项目

把交付 ZIP 上传服务器并解压，得到 `AIGI01_overlay/`。无需先 clone 空仓库。示例假设包放在 `$HOME/downloads`，目标项目放在 `$HOME/projects/AIGI01`。

```bash
cd "$HOME/downloads"
unzip AIGI01_additive_project.zip
python3 AIGI01_overlay/aigi_scripts/bootstrap_upstream.py \
  --destination "$HOME/projects/AIGI01"

python3 AIGI01_overlay/aigi_scripts/bootstrap_upstream.py \
  --destination "$HOME/projects/AIGI01" \
  --remote https://github.com/nadaojiudaifei/AIGI01.git \
  --author-name "Your Name" --author-email "your-git-email@example.com" \
  --execute --push
```

脚本先检查远程确实为空，再获取固定上游提交，创建 `main`，先推送未加入任何扩展的原 DiT-IC。第一次推送成功后，才复制增量目录、检查原文件、提交扩展并第二次推送。

遇到 403 请在 GitHub 的连接/应用安装设置和仓库访问范围中检查 `AIGI01` 的代码写权限，或使用你自己在服务器上已获授权的 Git 凭据。不要把 token 粘进项目。不建议为了绕过错误强制推送或删除已有内容。

暂时只生成本地项目时，去掉 `--push`。之后手动上传仍可按先原代码、后扩展的顺序执行：

```bash
cd "$HOME/projects/AIGI01"
git ls-remote origin
# 仅当上一条输出为空，才按以下方式初始化空远程：
git push origin cd43f5d9761fb34f5224622145629d3ff2b89ca1:refs/heads/main
git push -u origin main
```

如果目标目录已经非空，导入脚本会拒绝覆盖。不要在已有项目内反复运行 `git init` 或执行清空目录的命令。

## 2. 在远程创建 Conda 环境

服务器已经有 Conda 时直接使用它。还没有 Conda 时，从官方 Miniconda 下载页取得与你服务器架构一致的 Linux 安装包，核对官方 SHA256，再执行安装程序；不要使用来历不明的一键脚本。官方入口见 `SOURCES.md`。

```bash
cd "$HOME/projects/AIGI01"
conda env create -f aigi_environment.yml
conda activate aigi01
python --version
# 预期 Python 3.12.x

python -m pip install --upgrade pip
python -m pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r aigi_requirements.txt
python -m pip check
```

**不要直接执行原来的 `pip install -r requirements.txt`。** 原文件保留不动，但它指定的 NumPy 2.4.3 与 CompressAI 1.2.8 的 `numpy<2` 约束冲突。新增文件固定 NumPy 1.26.4，并限制 OpenCV 不进入要求 NumPy 2 的版本。原 PyIQA 0.1.14.1 又固定 Transformers 4.37.2；本工程需要 Gemma2，因而选择 PyIQA 0.1.15.post2 和 Transformers 4.55.4。指标环境有变化，后续必须在这套同一环境内重测基线。

新依赖清单是明确的兼容目标，不是本次已完成安装解析的锁文件。以 `pip check`、`doctor` 和远程验收为准。不要为强行安装而使用 `--no-deps`。CompressAI Linux 预编译轮子可能要求较新的 glibc；旧 Linux 发行版需要管理员提供兼容系统或构建工具。

```bash
export PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HOME="$HOME/cache/huggingface"
export TORCH_HOME="$HOME/cache/torch"
mkdir -p "$HF_HOME" "$TORCH_HOME"
python -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
python -m aigi --help
python -m aigi catalog
python -m aigi doctor --require-gpu --output aigi_runs/doctor_environment.json
bash aigi_scripts/check_cpu.sh
```

`doctor` 返回非零退出码表示缺版本、GPU 或文件，不会自行安装任何东西。CPU 单元测试通过仍不等于真实模型已验收。

## 3. 下载模型，或者使用现有模型

先查看计划。`models` 不加 `--download` 只查询文件与大小，不开始大规模下载：

```bash
python -m aigi models --output aigi_assets --qualities 3
python -m aigi models --output aigi_assets --qualities 3 --download
# 正式做四个质量点时：
python -m aigi models --output aigi_assets --qualities 1 2 3 4 --download
```

下载内容是 SANA-600M Diffusers、DiT-IC 官方指定质量点 merged 权重、ELIC 辅助编码器和 BLIP caption 模型。记录文件为 `aigi_assets/assets.lock.json`，包含解析到的 HF commit 和文件清单。大模型不放入 Git。

本地已有相同模型时，不必重新下载。按照实际位置配置即可。推荐保留 SANA 的 `transformer/`、`vae/`、`text_encoder/`、`tokenizer/`、`scheduler/` 子目录，以及原始配置与 safetensors 文件：

```bash
python -m aigi configure --base aigi_configs/joint_q3.json \
  --output aigi_configs/local_joint.json \
  --set 'sana_path=/data/models/sana' \
  --set 'weights=/data/models/ditic/q3_merge_ema.pt' \
  --set 'elic_path=/data/models/ditic/elic_official.pth' \
  --set 'offline=true'
```

使用本工程下载的模型时，命令会查找 ELIC 和质量点权重真实子路径：

```bash
python -m aigi configure --base aigi_configs/joint_q3.json \
  --assets aigi_assets --output aigi_configs/local_joint.json
```

默认 `model_variant=fp16` 指文件命名变体，**运行计算仍是 FP32**。你的本地目录只有不带 variant 后缀的原始权重时，明确设置 `--set 'model_variant=""'`。不要混合不同 SANA 尺寸、1024/512 分支或不同架构的 checkpoint。

GitHub Release 或其他可信 HTTPS 上已有模型，必须提供发布方核实过的 SHA256：

```bash
python -m aigi model-url \
  --url 'https://github.com/OWNER/REPO/releases/download/TAG/weights.pt' \
  --output '/data/models/weights.pt' --sha256 'REPLACE_WITH_64_HEX_DIGEST'
```

这只是通用权重下载器，不会把任意模型转换成 SANA/DiT-IC 格式。只加载可信 tensor checkpoint；程序不自动回退到不安全 pickle 加载。

## 4. 准备小数据集，先不要下载 TB 级全量数据

自然图像训练起步：

```bash
python -m aigi data-hf div2k_train --output aigi_data/div2k_train --limit 200
python -m aigi data-hf kodak --output aigi_data/kodak
```

AI 图像起步选取 DiffusionDB 官方第一个 1000 图归档：

```bash
python -m aigi data-hf diffusiondb_1k --output aigi_data/diffusiondb_1k
```

已有下载文件时：

```bash
# 官方 DiffusionDB part-000001.zip 可以位于目录根部或 images/ 中。
python -m aigi data-hf diffusiondb_1k \
  --local-source /data/diffusiondb_archives --output aigi_data/diffusiondb_1k

# Hugging Face save_to_disk / split parquet / imagefolder 本地副本：
python -m aigi data-hf sdxl10k --local-source /data/sdxl10k \
  --output aigi_data/sdxl10k --split train --limit 200

# 普通图片文件夹，支持同名 .txt、metadata.jsonl 或 DiffusionDB part JSON。
python -m aigi data-local --root /data/my_images \
  --output aigi_data/my_images.jsonl --dataset my_images --kind aigi
```

不要把普通文件夹当作 Hugging Face 缓存目录。现有数据就是 PNG/JPEG 文件夹时，优先 `data-local`。导出清单保存绝对图片路径、文件 SHA256、像素 SHA256、原始尺寸、prompt 来源和 split，不在评测前缩放图像。

自然图像通常没有 prompt。SDXL10k 数据卡也没有提供原始 prompt；为缺失项生成描述：

```bash
python -m aigi caption --manifest aigi_data/div2k_train/manifest.jsonl \
  --output aigi_data/div2k_train/captioned.jsonl --model aigi_assets/caption
```

BLIP 描述只填充缺失项，不覆盖原始生成提示词。它是替代描述，不应在论文中称为“原始生成 prompt”。

```bash
python -m aigi split \
  --inputs aigi_data/div2k_train/captioned.jsonl aigi_data/diffusiondb_1k/manifest.jsonl \
  --output aigi_data/splits --seed 903
cat aigi_data/splits/split_report.json
```

默认按连通组做 98%/1%/1% 划分。相同像素或相同原始 prompt 会合并成同组；官方 validation/test 永远不转成训练。小到只有几张图时可能没有验证样本，程序会报错，此时用足够多样本，或分别准备真正独立的 train/validation 清单，不要把训练图复制一份冒充验证图。Kodak 始终单独评测，不放入训练输入。

## 5. 写配置、检查资源、预热指标

```bash
python -m aigi configure --base aigi_configs/local_joint.json \
  --output aigi_configs/local_joint.json \
  --set 'train_manifest=aigi_data/splits/train.jsonl' \
  --set 'val_manifest=aigi_data/splits/val.jsonl' \
  --set 'teacher_cache=aigi_cache/teacher' \
  --set 'workers=2' --set 'batch_size=1' --set 'accumulation=1'
python -m aigi doctor --config aigi_configs/local_joint.json \
  --require-gpu --output aigi_runs/doctor_full.json
python -m aigi warmup --device cuda
```

`warmup` 是显式联网预热 LPIPS、DISTS、NIQE、CLIPIQA、MUSIQ、FID/KID 权重。后续离线运行应保持同一个 `HF_HOME` / `TORCH_HOME`。默认训练和评测不隐式下载权重，缺少缓存会失败。

## 6. 教师缓存：训练前必须做

教师不是每个训练 step 重跑，而是先为确定的图片、prompt、质量点、裁剪尺寸建立缓存。默认每图需要 8 个 log-SNR 节点 × 2 个噪声样本 × 条件/空条件各一次，即 32 次教师 DiT 前向。对 2M/14M 数据盲目运行会产生很大的计算与存储成本，先验收小数据集。

```bash
python -m aigi teacher --config aigi_configs/local_joint.json \
  --manifest aigi_data/splits/train.jsonl
python -m aigi teacher --config aigi_configs/local_joint.json \
  --manifest aigi_data/splits/val.jsonl
# 之后要在测试集运行推理不需要教师缓存。
```

如果使用你额外生成的 prompt 清单，请一直使用那份清单；改图片、prompt、quality 或 crop 后会产生新缓存 key。不同教师模型不能混用同一缓存目录，`cache_contract.json` 会校验。不要同时让多个进程写同一个缓存目录；当前教师导出是顺序执行，不是分布式预处理器。

## 7. 先运行远程最小验收

```bash
# 只打印计划。
bash aigi_scripts/remote_acceptance.sh aigi_configs/local_joint.json aigi_runs/acceptance
# 明确执行：每种方法 2 个优化步，真实编码，两个独立进程解码。
bash aigi_scripts/remote_acceptance.sh aigi_configs/local_joint.json aigi_runs/acceptance --execute
```

验收要求已有 train/val 数据与模型。它会为选出的3张图单独建教师缓存，所以也可在全量教师导出之前先做这一步。测试使用真实依赖，LPIPS/DISTS/GAN 暂时关闭；不会伪造模型替代运行。最终 `acceptance.json` 只有所有命令成功才写 passed。V2 默认验收还比较“完整两步训练”与“第一步中断后恢复”的精确状态。
`python -m aigi acceptance ... --extended --gpus 2 --execute` 可增加 GAN 累积、指标加载和两卡短训练；
其中小数据不足以统计 FID/KID，会保留不可用原因。长期收敛、完整大数据结果和大图效率仍要另跑。

## 8. 单独训练地图与完整新方法

可选的地图预训练用于单独测试学生地图模块；不是强制步骤：

```bash
python -m aigi configure --base aigi_configs/local_joint.json \
  --output aigi_configs/local_maps.json --set 'stage=maps' \
  --set 'steps=10000' --set 'output=aigi_runs/maps_q3'
bash aigi_scripts/train_single.sh aigi_configs/local_maps.json

# 使用地图 checkpoint 初始化完整模型。
python -m aigi configure --base aigi_configs/local_joint.json \
  --output aigi_configs/local_joint.json \
  --set 'maps_checkpoint=aigi_runs/maps_q3/last.pt'
```

完整训练默认从官方 q3 merged DiT-IC 权重继续训练新增组件与 LoRA，而非随机初始化整套模型：

```bash
bash aigi_scripts/train_single.sh aigi_configs/local_joint.json
# 或直接运行，追加 --resume 可完整恢复本训练器 checkpoint：
python -m aigi train --config aigi_configs/local_joint.json \
  --resume aigi_runs/joint_q3/last.pt
```

输出包括 `last.pt`、`resolved_config.json`、逐步训练日志和 `validation_smoke.json`。最后者只检查一张确定裁剪的验证图，不是整个验证集指标。恢复要求相同的训练配置、step 预算和 DDP 卡数；改变 loss、分辨率、质量点等应开启新阶段，用 `init_checkpoint` 加载权重而不是用 `resume`。

仅使用预训练 SANA/ELIC、随机初始化主压缩器开始训练时：

```bash
python -m aigi configure --base aigi_configs/local_joint.json \
  --output aigi_configs/from_base.json --set 'checkpoint_format=base' \
  --set 'maps_checkpoint=""' --set 'output=aigi_runs/from_base_q3'
python -m aigi train --config aigi_configs/from_base.json
```

这不是“从零训练 SANA”。随机 DiT 消融必须显式 `dit_from_scratch=true` 且 `finetune=full`，否则程序拒绝。

## 9. 多卡与第二阶段

```bash
# GPU 数量是脚本第一个参数。
CUDA_VISIBLE_DEVICES=0,1 bash aigi_scripts/train_ddp.sh 2 aigi_configs/local_joint.json
```

有效全局 batch = 每卡 batch_size × GPU 数 × accumulation。减少显存首先减每卡 batch，用 accumulation 保持有效 batch。本实现固定 FP32，不要临时加自动混合精度然后仍使用原有 entropy 可重复性结论。梯度检查点重计算被关闭，因为局部 norm hooks 只在一次前向作用域内安装。

第二阶段 crop=512 时必须重建 512 的 train/val 教师缓存：

```bash
python -m aigi configure --base aigi_configs/local_joint.json \
  --output aigi_configs/local_joint_stage2.json --set 'crop=512' \
  --set 'steps=60000' --set 'gan_weight=0.1' \
  --set 'init_checkpoint=aigi_runs/joint_q3/last.pt' \
  --set 'maps_checkpoint=""' --set 'output=aigi_runs/joint_q3_stage2'
python -m aigi teacher --config aigi_configs/local_joint_stage2.json --manifest aigi_data/splits/train.jsonl
python -m aigi teacher --config aigi_configs/local_joint_stage2.json --manifest aigi_data/splits/val.jsonl
python -m aigi train --config aigi_configs/local_joint_stage2.json
```

## 10. 推理与真正独立的解码

整个清单的编码、解码及总码率记录：

```bash
python -m aigi infer --config aigi_configs/local_joint_stage2.json \
  --checkpoint aigi_runs/joint_q3_stage2/last.pt --ema \
  --manifest aigi_data/kodak/manifest.jsonl --output aigi_runs/kodak_joint_q3
```

Kodak 清单没有 prompt 时，以上等价于使用空 prompt。要评价基于描述的条件方法，先对 Kodak 生成独立 captioned 清单，并将同一清单用于所有新方法；原 DiT-IC 忽略外部 prompt。总码率会包含描述字节，不能免费省掉。

单图分离编码与解码：

```bash
printf '%s' 'A dog running on grass.' > /tmp/prompt.txt
python -m aigi encode --config aigi_configs/local_joint_stage2.json \
  --checkpoint aigi_runs/joint_q3_stage2/last.pt --ema \
  --input /data/example.png --prompt-file /tmp/prompt.txt --output /tmp/example.aigi
python -m aigi inspect --input /tmp/example.aigi

# 可以放到另一台具有相同模型、代码和数值环境的服务器；不传原图或教师缓存。
python -m aigi decode --config aigi_configs/local_joint_stage2.json \
  --checkpoint aigi_runs/joint_q3_stage2/last.pt --ema \
  --input /tmp/example.aigi --output /tmp/reconstructed.png
```

解码器的接口没有原图参数。默认 prompt 存在码流内；`--external-prompt` 只用于另一种实验假设，此时解码也必须提供完全一致的 `--prompt-file`，空格/换行变化都会触发哈希不匹配。编码与解码的 EMA 选择也必须一致。

## 11. 评测与矩阵

```bash
python -m aigi evaluate --input aigi_runs/kodak_joint_q3/inference.json \
  --output aigi_runs/kodak_joint_q3/metrics.json \
  --metrics psnr ms_ssim lpips dists niqe clipiqa musiq

python -m aigi matrix --config aigi_configs/local_joint.json \
  --output aigi_runs/matrix_small --qualities 3 --no-ablations --no-add-one \
  --datasets aigi_data/kodak/manifest.jsonl
python -m aigi run --plan aigi_runs/matrix_small/matrix.json
# 核对计划、数据/权重/256和512教师缓存齐备后才执行：
python -m aigi run --plan aigi_runs/matrix_small/matrix.json --execute
```

默认完整矩阵可能产生数百个长训练任务，不会自动运行。矩阵不自动准备教师缓存，四质量点、两个裁剪尺寸必须分别准备。失败任务不会被标成成功；运行器保存退出码与日志。更多消融、四点 BD-rate、盲测和原论文覆盖差异见 `EXPERIMENTS_ZH.md`。

## 12. 常见故障

`datasets` 名称冲突：HF 导出使用隔离子进程；不要在原项目根目录手动写 `from datasets import load_dataset` 并期待一定导入 HF 库。普通图像清单训练不调用 HF datasets。

缺少 cache / 哈希不匹配：重新核对清单、crop、quality、prompt、教师路径。不要关闭校验或用旧缓存改名糊弄。

CUDA OOM：降低每卡 batch，先使用 256 crop 与较低 LoRA rank 的独立实验。降低 rank 之后需要重新训练，不能直接严格加载另一个 rank 的训练 checkpoint。真实大图解码是否可用以 `profile` 和显存实测为准。

指标缺权重：联网运行 `warmup`，确认离线时使用同一 cache 目录。不要用 0 或空字符串代替失败的指标。

训练命令不认识参数：执行 `python -m aigi COMMAND --help`。本指南里的所有主命令均由同一个入口实现，不需要修改原训练脚本。