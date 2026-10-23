# 实际训练及数据处理

本提交继承本队早期 V7/v500 任务训练结果，v972 从 v500 续训 1000 steps，v973 从 v972 续训 400 steps，选用 v973 step 200 的 best task 权重，不是从官方预训练模型仅训练 200 步得到的方案。`original/` 保留实际预处理、训练实现及可取得的阶段配置；`initial/best_task_heads.pt` 是最后训练阶段所需的 v972 初始化任务头，仅供训练，不加载到正式推理中。

v972、v973 使用 `no_lm` 和 `freeze_season`，冻结 LoRA 与内部季节头，只续训影像解码器。v972 与 v973 的 LoRA 文件 SHA-256 相同，因此复现最后阶段时复用提交的 `weights/v973_adapter/`。早期训练中的蒸馏监督属于训练过程，正式入口不加载教师或外置 season 分类器。模型没有量化或跨架构权重转换，推理时由 PEFT 将 LoRA 合并入同一 BF16 基座。

## 官方数据与预处理

将组委会发布的官方数据放在一个工作根目录下：`train/`、`val/` 中保留各自光学、SAR、SCL、MASK 和 QA，`seg_data/MASK_逐影像类别解释.csv` 保留官方逐影像类别解释。程序按题目影像引用匹配同一影像的 MASK 和 SCL，不从测试题读取训练标签。

```bash
dependencies/runtime/bin/python3 training/prepare_data.py --work-root /path/to/official_work
dependencies/runtime/bin/python3 training/train_stage.py \
  --work-root /path/to/official_work \
  --model /path/to/B榜材料/weights \
  --output /path/to/reproduced_v973
```

`prepare_data.py` 调用归档的实际预处理函数并只覆盖目录参数，依次完成：

1. `train_area_qa_patch_v2.py`：光学除以 10000，SAR 按 `(x+35)/40` 归一化，附加光学/SAR 有效性，面积依据官方 MASK、有效像元和逐影像类别解释计算；生成 256×256 的 16 通道缓存。
2. `build_code_cache.py`：生成七类码级监督、五类面积和官方训练季节标注；码 24/26 按官方逐影像解释映射。
3. `build_scl_cache.py`：追加 SCL/9 及类别 3/8/9 的云影指示，得到 18 通道缓存；缺失 SCL 置零。
4. `build_y7_500.py`：MASK 和有效像元按 `[::2, ::2]` 得到 500×500 分割监督；连续面积标签仍按原始分辨率计算。

中间文件为 `area_qa_patch_v2/manifest.json`、`cache_train|val/*.npz`、`code_cache/`、`code_cache_scl/` 和 `y7_500/`。NPZ 包含 `x`、`y7`、`areas`、`season` 等实际训练成员；高分辨率监督包含 `cdl`、`valid`。它们均从官方数据生成，不包含外部数据集。

## 配置及校准

`train_stage.py` 为实际 v973 训练实现提供相对目录入口：400 steps、group=32、micro=32、lr_head=5e-5、reweight=0.5、DEC_RES=500、无几何增强、每 200 步评估和留档。训练混合使用 2697 张官方训练侧影像及 899 张官方验证侧影像，影像键分别保留在 `original/v972_train_pool.json` 和 `original/v972_val_keys.json`。该阶段评估沿用了参与训练的验证侧集合，历史 task 不是严格独立的泛化分数。其他参数及原始归一化统计在 `original/v973/config.json`；归档配置中的服务器路径是历史记录，当前入口通过参数指定目录。

本包校准对应官方训练池之外的 1433 张影像；继承的早期任务权重可能见过其中影像，不能视为整个模型历史严格未见的数据。`calibrate_v973_unseen_t2.py`、`calibrate_v973_unseen_t3.py` 和 `make_v973_unseen_calibration_keys.py` 是实际拟合/选集代码；生成最终候选的两个 `make_v973_*candidate.py` 脚本保留固定系数的应用过程。最终部署只读根目录 `calibration.json`，T2 决策阈值固定 1.0 km²。未把 A/B 测试答案、历史预测文件或缓存答案加入推理包。

使用的官方比赛数据不重复作为外部资源打包；没有为此提交路线引入额外外部训练数据。Qwen 基座的原始预训练数据和代码按公告不要求提交。实际第三方依赖和模型来源、版本及许可证见根目录 README 与 `external_resources/resources.json`。
