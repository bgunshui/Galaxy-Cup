# 决赛 B 榜材料

本包对应 `candidate_v973_t3_scale_res500_t2_10.jsonl` 的推理方案。模型为 Qwen3-VL-8B-Instruct、v973 LoRA 与内部 18 通道影像解码器，作为一个 checkpoint 加载；不调用外置分类器、其他独立模型或在线服务。本包提供完整基座权重，不需要另行下载模型。

## 安装与推理

目标环境：Linux ARM64/aarch64、Jetson AGX Thor/T5000、CUDA 13.0。官方基础环境提供 NVIDIA 驱动和 CUDA 13.0；本包不重复携带 CUDA Toolkit、CUDA 驱动或 NVIDIA CUDA 运行库。包内只保留 CPython 3.10 ARM64 运行时、Torch/应用 wheel 和本队代码，无自定义 CUDA 算子。安装前需有平台提供的 NVIDIA 驱动、CUDA 13.0、Bash、tar 和 glibc 2.28 或以上。解压后的包路径及数据路径可以自行指定。

```bash
cd /path/to/B榜材料
bash dependencies/install_offline.sh
bash run_inference.sh \
  --input /path/to/qa_test.jsonl \
  --data-root /path/to/test_data \
  --model /path/to/B榜材料/weights \
  --output /path/to/result.jsonl
```

安装脚本使用随包的 ARM64 Python 和应用 wheel，通过 `--no-index --no-deps` 安装，不访问网络；CUDA 依赖由官方基础环境提供。安装不计入正式推理时间。推理入口关闭 Hugging Face 在线下载，只读取本地资源；输出为 UTF-8 JSONL，保持输入题目顺序，`answer` 为 JSON 对象字符串。

`--data-root` 指向影像所在数据根目录，可包含 `test/` 子目录；读取题目 `images` 中对应的光学、SAR 文件，支持公告数据的中文及 `#Uxxxx` 转义目录名。正式入口固定采用历史候选的 SCL 缺失路径，两个 SCL 通道以训练均值补齐，归一化后为零，不读取测试侧 SCL。

模型在 CPU 上加载 BF16 基座并合并本地 LoRA，然后送入 GPU，安装内部任务头。每批 8 题，同一次模型执行产生语言答案和内部面积向量；影像输入为 256，面积解码器分辨率为 500，最多生成 48 tokens。正式入口不进行多尺度、翻转 TTA 或多模型融合。

## 文件及资源清单

| 本地路径 | 内容及用途 | 来源与版本 |
|---|---|---|
| `weights/base/` | 完整 BF16 基座、4 个权重分片、配置、tokenizer、processor | 官方 Qwen3-VL-8B-Instruct，固定版本见下文 |
| `weights/v973_adapter/` | 当前 checkpoint 的 LoRA 权重与配置 | 本队在官方比赛数据上微调 |
| `weights/v973/` | 当前 checkpoint 的影像任务头及加载配置 | 本队 v973，best task 为 step 200 |
| `inference/`、`run_inference.sh` | 影像读取、统一模型执行、答案解析与写出 | 本队实际推理代码及可移植入口 |
| `calibration.json` | 固定的作物面积校准系数、T2 阈值 | 官方训练数据拟合，T2 为 1.0 km² |
| `requirements.txt`、`dependencies/` | 官方基础环境版本清单、ARM64 Python、应用 wheel、安装脚本 | 不含 CUDA Toolkit/驱动；版本、原始来源及许可证见 `external_resources/resources.json` |
| `training/` | 实际训练、预处理、校准代码，关键配置与上阶段初始化任务头 | 使用说明见 `training/README.md` |
| `external_resources/` | 第三方资源来源、固定版本、许可证 | Qwen Apache-2.0；依赖许可证在资源清单和 wheel 内 |
| `SHA256SUMS` | 所有提交文件的 SHA-256，清单本身除外 | 解压后可用 `sha256sum -c SHA256SUMS` 校验 |

上述资源实际通过本压缩包内的本地文件交付；下面的地址是原始来源，不是需要评测方额外寻找的缺失材料。

- 基座原始来源：<https://modelscope.cn/models/Qwen/Qwen3-VL-8B-Instruct>，revision `1610d9f98460d59272cd9963efdf92f0cd3c8dc9`，公开、匿名可访问。按固定版本获取的全部推理文件已随包提供；逐文件版本、字节数和官方 SHA-256 见 `external_resources/resources.json`。上游授权 Apache-2.0，全文见 `external_resources/Qwen-LICENSE.txt`。
- 官方比赛训练数据由组委会发布，含光学、SAR、SCL、MASK、QA 和类别解释 CSV。未额外使用外部数据微调该提交路线，不重复打包官方原始数据；基座原始预训练数据按公告豁免。处理格式及实际代码见 `training/`。
- 本队代码及任务权重用于本次赛事部署和组委会核验；第三方组件保留各自许可证，原始来源、版本和交付位置明确列在资源清单中。
- CUDA 驱动、CUDA Toolkit 13.0 和 NVIDIA CUDA 运行库由评测基础环境提供，不作为提交资源重复交付。

## 固定输出规则与验证

T1 取内部面积向量最大类别；T2 对同一向量进行训练集拟合的作物校准后，严格使用 `>= 1.0 km²`；T3 使用同一模型生成的数值并执行固定作物系数；T4 解析该模型生成的选项；T5 使用同一内部面积向量和题目比较规则。T3/T4 的非法生成回退也仅使用该 checkpoint 的内部输出。具体常数保存在 `calibration.json`，不按 B 榜数据重新拟合。

校准样本来自最新 v972/v973 训练池之外的官方训练影像，但继承的更早任务权重可能见过这些影像，因此不把该集合宣称为全训练历史严格未见的验证集。未读取测试答案，也不按测试编号或文件名预设答案。

在 RTX 4090 48 GiB 上，开启禁止 IP socket 连接的检查后，完整 2000 题从加载到输出耗时 263 秒，与对应历史候选的 2000 条答案逐条语义一致，解析回退为 0。本记录是 4090 实测，不是 Thor 实测；ARM64 依赖已按目标平台准备，Thor 的实际驱动兼容性和全量耗时仍需在组委会硬件上验收。

技术联系人：本参赛队伍在赛事平台登记的技术联系人，沿用参赛登记的联系电话与邮箱，可通过赛事平台联系本队。
