# 验证报告与未完成事项

日期：2026-09-28。本文是实际执行记录，不把计划或测试替身视为生产验证。

## 1. GitHub 与环境状态

目标仓库：`nadaojiudaifei/AIGI01`。本次连接能读取仓库，但新增工作流文件和新增普通源码文件两次写入均返回 `403 Resource not accessible by integration`。**没有成功创建远程提交，没有把原 DiT-IC 上传到该仓库，也没有把本增量代码上传到该仓库。**

交付的是可下载增量源码包及服务器端 bootstrap 脚本。脚本设计为先推送纯上游固定提交，后推送增量；该真实网络推送未在本次会话执行。不能将已经写好这个脚本描述成已经完成 GitHub 上传。

本次没有创建/激活新的 Conda 环境，没有安装软件包，没有下载模型权重或训练数据。使用的是当前会话已有 CPU Python/PyTorch。精确版本与平台见 `validation.json` 和 `local_preflight.json`。

## 2. 实际执行通过的检查

最终命令：

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -c aigi_pytest.ini -q \
  --junitxml=aigi_docs/cpu_tests.xml
python -m compileall -q aigi aigi_scripts aigi_tests
for f in aigi_scripts/*.sh; do bash -n "$f"; done
python -m aigi --help
python -m aigi catalog
```

结果：**70 项 CPU 测试通过**，没有把 GPU 测试标成通过。原始输出见 `cpu_tests.txt`，JUnit 记录见 `cpu_tests.xml`。7份配置均通过严格配置解析；23个CLI子命令的help可以生成；所有新增Python文件可解析/编译；所有新增Shell脚本通过bash语法检查。

测试内容包括：STE梯度，归一化常量图，重要性映射，流速度转噪声恒等式，c/e积分和负e保留，量化边界，SNR和时间单调性，双路引导代数，water-filling预算守恒，归一化加权误差，padding，文本mask，token权重，局部norm hooks移除，配置和消融生成，数据哈希与传递去重，码流封装/外部prompt/损坏拒绝，checkpoint与RNG保存恢复，EMA，BD-rate边界，以及原文件哈希保护的本地Git小型夹具。

## 3. 哪些测试使用了替身

`test_causal_loop.py` 中的概率表、符号记录器、微型骨干是**仅存在于测试代码中的替身**。其作用是检查四组编码/解码共享顺序、解码接口没有原图参数、不同分支NFE、hook清理和基础计算图。

这些替身不是真正 CompressAI rANS，不使用真实 SANA 权重，也不复刻全部原 checkerboard mask 的数值行为。生产代码没有这些替代分支。缺少真实依赖时，正式训练/编解码会失败而不是生成一份假的“成功结果”。

由此可以说“接口和部分数学逻辑通过测试”，不能说“真实熵编码往返和全模型训练通过测试”。

## 4. 本地预检为何返回非零

实际执行：

```bash
python -m aigi doctor --output aigi_docs/local_preflight.json
```

退出码为 **1**。这是预期且被如实保留的结果：本会话没有CUDA GPU，缺少CompressAI/Diffusers/Transformers等目标依赖，已有PyTorch/NumPy也不是远程目标版本。本次没有为了把报告变绿而安装环境、降低目标要求或篡改结果。

`doctor`只是只读检查，不是安装器，也不是模型集成测试。

## 5. 尚未执行的远程验证

以下项目有实现入口，但本次没有执行到真实系统通过：HF数据导出与模型下载，真实merged/LoRA权重严格加载，真实SANA教师缓存，真实rANS编码/解码，两个独立进程的全模型重建一致性，单卡优化与恢复，多卡DDP，GAN阶段，全部感知指标权重初始化，大图显存/性能，长训练稳定性，以及率失真或跨生成器改进。

运行入口：

```bash
bash aigi_scripts/remote_acceptance.sh aigi_configs/local_joint.json aigi_runs/acceptance
bash aigi_scripts/remote_acceptance.sh aigi_configs/local_joint.json aigi_runs/acceptance --execute
```

验收会单独准备两张训练图和一张验证图的教师缓存，对五种方法分别进行2个优化步、真实编码和两个新进程解码，最后比较两次PNG哈希。只有全部成功才写 `passed=true`。这个有限smoke gate也不证明收敛、GAN、多卡或论文性能。

## 6. 本版尚未实现的需求

OCRBench V2任务评估、原论文完整分模块FLOPs/计时表、真实人工研究的统计汇总尚未实现。完整CADC模型不是本版对照实现，只有明确命名的CADC-style迁移项。原论文150K精选训练集合没有被自动重建，原训练器的AdamW、随机增强和自适应GAN细节也没有全部复制到新增训练器。

因此用户最初要求的“仓库已经落地 + 所有原实验全覆盖 + 所有功能正常执行保证”在本次交付中**尚未全部完成**。已完成部分是增量代码、明确算法、可运行入口、配置、中文指南与上述CPU验证；其余没有隐藏在含糊的“支持”表述中。

## 7. 建议的验收顺序

先解决GitHub写权限并导入，再在远程安装固定目标环境并运行pip check/doctor；随后用小数据执行remote acceptance，再单独跑GAN、DDP、完整指标与checkpoint恢复。最后才扩大数据、分辨率、质量点与消融矩阵。

任何一步失败都应保留stderr、环境版本和失败配置，修复后重新执行对应验收。不能将失败样本删除后仍宣称整个数据集成功，也不能用CPU接口测试替代真实GPU测试。