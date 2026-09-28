# 数据集、模型、本地与下载模式

登记项见 `aigi/registry.py`，执行 `python -m aigi catalog` 可打印机器可读清单。以下规模来自数据卡/仓库，不是本次已下载并验证过的数量。下载时保存解析到的 revision 和实际导出数量；镜像更新、列名变化、权限变化都会使命令明确失败，不会默默换另一个数据集。

## 1. 自然图像

| CLI 名称 | Hugging Face 仓库 | 规模/用途 | 注意事项 |
|---|---|---|---|
| kodak | danjacobellis/kodak | validation 24图，小型标准测试 | 仅评测，不用于训练 |
| div2k_train | yangtao9009/DIV2K | train 800图，小型高质量训练 | 不是 DIV2K validation 100图 |
| lsdir | danjacobellis/LSDIR | 约85K图，大型自然图训练 | 不等于作者训练所选约50K子集 |
| mlic100k | Whiteboat/MLIC-Train-100K | 约100K图，大型训练 | 分卷7z，需完整下载并解压，不能用--limit减少归档下载量 |

Kodak 镜像公开显示 `image` 列和 `validation` split。DIV2K 镜像只有 train800；网页 viewer 出现大 row-group 的显示限制时，不等于原始 imagefolder 不能加载，但本次未实际执行下载。遇到数据源加载错误，请用已取得的官方图片配合 `data-local`，不能把失败当作成功导出。

```bash
python -m aigi data-hf kodak --output aigi_data/kodak
python -m aigi data-hf div2k_train --output aigi_data/div2k_train --limit 200
python -m aigi data-hf lsdir --output aigi_data/lsdir_small --limit 1000

# 大型下载必须明确确认资源预算。
python -m aigi data-hf mlic100k --output aigi_data/mlic100k --allow-large
```

MLIC 镜像采用37个分卷，合计约158GB的量级，解压后另占空间；以 `models`/HF 文件页和实际磁盘预算为准。`mlic100k --local-source` 接受已存在的分卷目录；普通已经解压好的图像文件夹应直接 `data-local`。

正式复现原测试集时，官方 DIV2K validation 与 CLIC2020 test 可以这样接入：

```bash
python -m aigi data-local --root /data/DIV2K_valid_HR \
  --output aigi_data/div2k_val.jsonl --dataset DIV2K_validation \
  --split validation --expected 100
python -m aigi data-local --root /data/CLIC2020_test \
  --output aigi_data/clic2020_test.jsonl --dataset CLIC2020_test \
  --split test --expected 428
```

`--expected` 只核对图片数，不证明这些图真的来自指定官方测试集。请保存官方来源、压缩包校验值，避免混入低分辨率版本、其他年份、重复图或训练图。

## 2. AI 生成图像

| CLI 名称 | 仓库 | 规模/用途 | prompt 情况 |
|---|---|---|---|
| diffusiondb_1k | poloclub/diffusiondb，官方part000001 | 约1000图，最小功能验收 | 归档JSON含原始prompt |
| diffusiondb_2m | poloclub/diffusiondb，小版 | 2M图，大型训练 | 原始prompt，图像总量约1.6TB |
| diffusiondb_14m | poloclub/diffusiondb，大版 | 14M图，超大规模 | 包含2M小版，图像总量约6.5TB |
| diffusiondb_filtered | whosouravsharma/text-to-image-diffusiondb-2M | 数据页面约18,219条，较小子集 | 有prompt；仓库名字不是实际2M条的证明 |
| sdxl10k | ash12321/sdxl-generated-10k | 10K张1024级SDXL图片 | 没有原始prompt列，需保留缺失或用替代caption |
| multi_generator | Shanmuk4622/ai-image-detection-dataset | 数据卡描述10K真实+六类各10K生成图 | 使用已有prompt/generator/label元数据 |

DiffusionDB 的第一份归档不是一个均匀代表完整14M分布的统计随机样本，适合跑通而非直接发表总体结论。14M包括2M，所以不能分别当作互不相交的训练/测试集。图片公开可见不表示内容均适合所有任务。

```bash
python -m aigi data-hf diffusiondb_1k --output aigi_data/ddb1k
python -m aigi data-hf diffusiondb_2m --first 1 --last 10 --output aigi_data/ddb10k
python -m aigi data-hf diffusiondb_filtered --output aigi_data/ddb_filtered --limit 1000
python -m aigi data-hf sdxl10k --output aigi_data/sdxl10k --limit 1000
python -m aigi data-hf multi_generator --output aigi_data/multi_generator --limit 1000

# 完整2M下载：只有明确资源预算后再执行。
python -m aigi data-hf diffusiondb_2m --output /data/diffusiondb_2m --allow-large
# 超大版可按归档号分段保存；大版归档路径由适配器选择。
python -m aigi data-hf diffusiondb_14m --first 10001 --last 10010 --output /data/ddb_large_segment
```

归档模式使用 `--first/--last`，不支持把 `--limit` 当作少下载几张图片。超过10份归档需要显式 `--allow-large`。当前归档导出按顺序执行，不是多机流式训练或断点分片数据服务。

## 3. 安全与数据质量规则

一般 HF 表格导出在存在明确 `image_nsfw` / `prompt_nsfw` 分数时使用0.1阈值过滤。官方 DiffusionDB 图像归档内部没有完整 NSFW 分数，本归档适配器不会虚构一个过滤结果；`export.json` 会提示这一点。需要完整内容筛选时先接入官方对应 metadata，做明确清洗，再生成研究清单。本版没有独立图像审核模型。

每个导出的 RGB 图记录文件 SHA256 和像素 SHA256。像素哈希用于识别同一图像以不同压缩格式保存的重复项；原始prompt经过 Unicode NFC、大小写折叠与空白规整后分组。使用连通分量处理“图片相同”和“prompt相同”的传递关系。

训练比例是连通组级98/1/1，不保证每个数据集正好98%。官方 test/validation 所在整个连通组被保留为评测组。这个设计偏重防泄漏，不是按类别严格分层抽样；要做类别均衡或按生成器留出，须明确构造对应清单。

## 4. 本地格式到底支持什么

普通图像目录支持 PNG/JPEG/WebP/BMP/TIFF，递归扫描。`image.ext` 的同名 `image.txt` 可提供 prompt；还支持 imagefolder 的 `metadata.jsonl`，以及 DiffusionDB `part-*.json` 中的 `p` / `prompt` 字段。

HF 本地副本支持 `save_to_disk` 目录、含 split 名称的 parquet 分片，以及合法 imagefolder。脚本在独立临时工作目录中以 `python -I` 运行 HF worker，避免原 DiT-IC 的 `datasets/` 目录抢占导入。

```bash
python -m aigi data-local --root /data/local_natural \
  --output aigi_data/local_natural.jsonl --dataset local_natural --kind natural
python -m aigi data-local --root /data/local_aigi \
  --output aigi_data/local_aigi.jsonl --dataset local_aigi --kind aigi
python -m aigi data-hf sdxl10k --local-source /data/sdxl_saved_dataset \
  --split train --output aigi_data/sdxl_export
```

导出得到 JSONL 清单，训练器只通过该清单读本地文件。它不是每个 batch 在线向 HF 拉图，因此网络波动不会在正式训练中悄悄改变数据。代价是你需要足够本地磁盘，并先完成导出。

路径为绝对路径时，搬到另一服务器需要同步目录结构或重新生成清单；改路径不应修改图片内容、prompt和分组。不要手工改 SHA256 以逃过被替换图像的检查。

## 5. 模型资产的固定约定

SANA：`Efficient-Large-Model/Sana_600M_1024px_diffusers`，指定600M/1024的 Diffusers 模型。需要 DC-VAE、28层SANA transformer、tokenizer、Gemma2 text_encoder，以及相应配置。下载时查询实际 revision 并保存。

DiT-IC：`JunqiShi/DiT-IC`。适配器明确查找 `q1_merge_ema.pt` 到 `q4_merge_ema.pt` 和 `elic_official.pth`，存在多个同名文件会报错，避免随机选错架构。

BLIP：`Salesforce/blip-image-captioning-base`，只用于缺失描述填充。使用固定 beam=3、最多64新tokens。它不是解码器的一部分，默认只在数据准备端运行；生成的文字按普通prompt计入码流。

`checkpoint_format=merged`：加载完整已合并DiT-IC权重；本工程训练时可再加新的LoRA。

`checkpoint_format=lora`：按原未合并LoRA字典格式加载，rank必须与来源一致；本次没有拿真实该格式权重执行加载验证。

`checkpoint_format=base`：使用SANA/ELIC预训练，主压缩模块初始化，不读取官方完整codec权重。训练完成后用自己的AIGI01 checkpoint推理。

本地/offline模式不会把网络仓库名当成本地目录自动下载。提供的目录缺文件时直接失败。依赖指标模型的缓存同样需要显式预热。

## 6. 许可证、可信下载与可重复性

数据镜像发布者、原始图像版权方、生成模型、下载平台的条款可能不同；请逐个阅读模型卡/数据卡。本文只登记可公开发现的资源，不为它们重新授权，也不对全部商业训练用途作法律结论。

本工程不执行 HF 的远程自定义 Python 数据集脚本；DiffusionDB旧式脚本型数据源走明确归档适配器。GitHub/HTTPS权重下载必须给SHA256，ZIP/7z拒绝目录穿越和链接条目。不将这些防护描述成对所有恶意输入的完整安全审计。

为了兼容原项目，本版固定 Diffusers 0.35.2，使用具体模型类和本地文件，不使用任意自定义 DiffusionPipeline。旧依赖存在已公开安全通告的可能，正式部署前应审核固定版本的官方通告并在升级后重新验收；不应为了复现而执行未知代码或加载不可信pickle。

来源链接与检索日期见 `SOURCES.md`。数据卡规模、HF split/列、下载服务可用性会变化，因此代码中的严格检查比一份静态说明更重要。
## V2 增加的 OCRBench 资产

`ocr-data` 使用 `ling99/OCRBench_v2` 的固定提交 `c7e7cdf23bdb6774661e9b0caf0d9935a42feb8b`，
可在线下载或用 `--local-source` 读取已经下载的 parquet/save_to_disk 目录。
保留完整 QA 字段，但压缩清单只含图像及空提示词。真实生成 caption 时只能依赖图像，不得读取问题或答案。

`ocr-assets` 明确下载 Qwen/Qwen2.5-VL-3B-Instruct 和固定 OCRBench 官方代码。
该 VLM 是本工程确定的统一评测器，不声称它就是原 DiT-IC 未明确公开的 OCR 模型。
HF 数据卡与上游仓库的数据用途说明不同：实际使用前自行核对原始数据集条件，本工程不替数据权利作保证。
详情与命令见 [V2 使用指南](V2_USAGE_ZH.md)。
