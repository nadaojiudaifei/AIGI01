# Primary sources and provenance

Research/metadata checked on 2026-09-28. No source below is evidence that our new method has been trained or has achieved an improvement.

## Code and papers

- DiT-IC official code: https://github.com/Eric-qi/DiT-IC
- Pinned source: https://github.com/Eric-qi/DiT-IC/tree/cd43f5d9761fb34f5224622145629d3ff2b89ca1
- DiT-IC paper: https://arxiv.org/html/2603.13162v1
- CADC paper (UGAQ scaling and no inverse scaling, Sections 4.1/4.2): https://arxiv.org/html/2602.21591v1
- SANA implementation: https://github.com/huggingface/diffusers/blob/v0.35.2/src/diffusers/models/transformers/sana_transformer.py
- Original metric implementation and patch extraction: https://github.com/Eric-qi/DiT-IC/tree/cd43f5d9761fb34f5224622145629d3ff2b89ca1/eval

The present cadc_style variant is NOT the full CADC implementation: it does not implement its complete ADGIC/BFATC pipeline. The proposal's finite teacher energies, local time residual, norm hooks, text entropy residual, and explicit training protocol are our engineering choices, not asserted findings of either source paper.

## Models

- https://huggingface.co/Efficient-Large-Model/Sana_600M_1024px_diffusers
- https://huggingface.co/JunqiShi/DiT-IC
- https://huggingface.co/Salesforce/blip-image-captioning-base

Downloads resolve and save exact HF revisions in assets.lock.json. This delivery contains no weights and does not claim to have validated those large downloads.

## Dataset cards

- https://huggingface.co/datasets/danjacobellis/kodak
- https://huggingface.co/datasets/yangtao9009/DIV2K
- https://huggingface.co/datasets/danjacobellis/LSDIR
- https://huggingface.co/datasets/Whiteboat/MLIC-Train-100K
- https://huggingface.co/datasets/poloclub/diffusiondb
- https://huggingface.co/datasets/whosouravsharma/text-to-image-diffusiondb-2M
- https://huggingface.co/datasets/ash12321/sdxl-generated-10k
- https://huggingface.co/datasets/Shanmuk4622/ai-image-detection-dataset

Official benchmark provenance:

- Kodak: https://r0k.us/graphics/kodak/
- DIV2K: https://data.vision.ee.ethz.ch/cvl/DIV2K/
- CLIC: https://archive.compression.cc/challenge/

Public mirrors are not automatically authoritative licensing sources. Preserve original provenance and check the actual files/splits. The filtered DiffusionDB repository name does not imply two million rows; SDXL10k does not supply original prompts; the 14M DiffusionDB includes the 2M subset.

## Dependency metadata and setup

- CompressAI 1.2.8 (requires numpy<2.0): https://pypi.org/pypi/compressai/1.2.8/json
- PyIQA 0.1.14.1 (pins transformers==4.37.2): https://pypi.org/pypi/pyiqa/0.1.14.1/json
- Selected PyIQA 0.1.15.post2 (transformers>=4.36.1): https://pypi.org/pypi/pyiqa/0.1.15.post2/json
- Diffusers 0.35.2 package/advisory metadata: https://pypi.org/pypi/diffusers/0.35.2/json
- PyTorch previous versions: https://pytorch.org/get-started/previous-versions/
- Miniconda: https://www.anaconda.com/docs/getting-started/miniconda/install
- HF local datasets: https://huggingface.co/docs/datasets/loading
- HF download/cache: https://huggingface.co/docs/huggingface_hub/guides/download

Changing PyIQA relative to upstream changes the measurement environment. Re-evaluate every controlled method under the same recorded environment rather than mixing published numbers and current numbers. The requirements file is an uninstalled target recipe, not a solver-tested lockfile. Concrete local model classes are used; arbitrary remote custom pipeline code and unsafe pickle fallback are not enabled.


## V2 sources (verified 2026-09-28)
- DiT-IC training / efficiency / OCR / human study: https://arxiv.org/html/2603.13162v1
- Official OCRBench V2 code, fixed commit: https://github.com/Yuliang-Liu/MultimodalOCR/tree/0ea56c2503a2d940700d581b777f23c16de0eafd/OCRBench_v2
- Official scoring interface: https://github.com/Yuliang-Liu/MultimodalOCR/blob/0ea56c2503a2d940700d581b777f23c16de0eafd/OCRBench_v2/README.md
- HF OCRBench V2 metadata: https://huggingface.co/datasets/ling99/OCRBench_v2
- Fixed HF export revision: https://huggingface.co/datasets/ling99/OCRBench_v2/tree/c7e7cdf23bdb6774661e9b0caf0d9935a42feb8b
- Fixed evaluator model family: https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct
- Compatible installed-version API: https://huggingface.co/docs/transformers/v4.55.4/model_doc/qwen2_5_vl
- Profiling limitations: https://docs.pytorch.org/docs/2.8/profiler.html
- Registered FLOP formulas and unsupported operators: https://github.com/pytorch/pytorch/blob/v2.8.0/torch/utils/flop_counter.py

The fixed Qwen evaluator, finite-view augmentation policy and participant-cluster statistical
protocol are engineering choices implemented here, not claims that DiT-IC used those exact details.
