# ColPali-Robust：视觉文档检索在退化场景下的鲁棒性分析

> **本项目基于 [illuin-tech/colpali](https://github.com/illuin-tech/colpali) 官方仓库构建**，在不修改任何原始代码的前提下，通过叠加研究层来分析 ColPali/ColQwen2 模型在真实恶劣场景下的检索能力。

---

## 研究背景与动机

ColPali 论文证明了"直接对文档页面图像做嵌入（Embedding）"的检索方式在高清干净 PDF 上效果极好。但在现实世界中——尤其是医疗档案、历史文献、金融发票等场景——文档图片往往存在以下问题：

| 退化类型 | 描述 | 现实场景 |
|---------|------|---------|
| 高斯噪声 | 随机像素扰动 | 低质量扫描仪、数字噪声 |
| 运动模糊 | 相机抖动导致的模糊 | 手机拍摄文件 |
| 图像倾斜 | 文档放置不正 | 扫描件、照片 |
| JPEG 压缩 | 过度压缩的伪影 | 网络传输、低质量存储 |
| 水印叠加 | 版权水印、机密标记 | 企业文档、法律文件 |

**核心问题：ColPali 对这些退化有多敏感？在检索前加入图像复原或背景分割，能否有效提升检索性能？**

---

## 研究方案（方案一：ColPali-Robust）

### 整体思路

```
原始 ColPali 流程：
  文档图像 → process_images() → 模型编码 → MaxSim 打分

ColPali-Robust 流程：
  文档图像 → [退化注入] → [复原/分割] → process_images() → 模型编码 → MaxSim 打分
                 ↑                ↑
              模拟现实          我们的改进
```

**关键洞察**：只需在 `process_images()` 调用之前插入一个预处理层，即可完整评估各类处理策略对检索性能的影响，**无需修改模型、无需重新训练**。

### 四个研究子任务

| 子任务 | 负责人 | 内容 |
|--------|--------|------|
| **A. 退化与复原** | 同学A | 构建退化数据集，用去噪/去模糊算法复原，量化复原效果 |
| **B. 文档分割** | 同学B | 用 Otsu 阈值+轮廓检测分割文档主体，去除无用背景 |
| **C. 域外泛化** | 同学C | 在训练集之外的极端文档类型上测试（中文古籍、手写笔记等） |
| **D. 评估与寻优** | 同学D | nDCG@5 评估框架 + PSO 粒子群算法寻找最优复原参数 |

---

## 项目结构

```
colpali-robust/
├── colpali_engine/              # 官方代码（未修改）
├── scripts/                     # 官方训练脚本（未修改）
│
├── robust/                      # 我们新增的研究包
│   ├── degradation/             # 图像退化模块（同学A）
│   │   ├── noise.py             # 高斯噪声、椒盐噪声
│   │   ├── blur.py              # 高斯模糊、运动模糊
│   │   ├── tilt.py              # 文档倾斜/歪斜
│   │   ├── jpeg.py              # JPEG 压缩退化
│   │   ├── watermark.py         # 水印叠加
│   │   └── pipeline.py          # 可组合的退化流水线
│   ├── restoration/             # 图像复原模块（同学A）
│   │   ├── denoise.py           # NLMeans 去噪、高斯平滑
│   │   ├── deblur.py            # Wiener 滤波去模糊
│   │   └── pipeline.py          # 可组合的复原流水线
│   ├── segmentation/            # 文档分割模块（同学B）
│   │   └── document_seg.py      # Otsu + 轮廓检测，背景置白
│   ├── evaluation/              # 评估指标（同学D）
│   │   └── metrics.py           # nDCG@K, Recall@K, MRR
│   └── optimization/            # 参数寻优（同学D）
│       └── pso.py               # PSO 粒子群优化器
│
├── experiments/                 # 实验脚本
│   ├── config.py                # 全局配置（模型、数据集、路径）
│   ├── run_benchmark.py         # 主实验入口（支持所有条件）
│   ├── pso_optimize.py          # PSO 寻找最优复原参数
│   └── visualize_results.py     # 生成对比图表
│
└── tests/                       # 单元测试（21 项，无需 GPU）
    ├── test_degradation.py
    ├── test_restoration.py
    ├── test_segmentation.py
    ├── test_metrics.py
    └── test_pso.py
```

---

## 环境配置

### 依赖安装

```bash
# 克隆本仓库
git clone https://github.com/w-yyh/colpali colpali-robust
cd colpali-robust

# 安装 colpali-engine（官方依赖）
pip install -e ".[train]"

# 安装额外研究依赖
pip install pyswarms opencv-python
```

### 硬件要求

| 配置 | 说明 |
|------|------|
| GPU 显存 | 建议 ≥ 8GB（ColQwen2-2B，bfloat16） |
| CUDA | 11.8+ |
| Python | 3.10+ |
| CPU 模式 | 修改 `experiments/config.py` 中 `DEVICE = "cpu"`（速度较慢） |

> **注**：如使用 Apple Silicon Mac，将 `DEVICE = "cuda"` 改为 `DEVICE = "mps"`

### 验证安装（无需 GPU）

```bash
python -m pytest tests/ -v
# 预期输出：21 passed
```

---

## 快速开始

### 1. 修改配置

编辑 `experiments/config.py`：

```python
MODEL_NAME = "vidore/colqwen2-v1.0"   # 可换成 "vidore/colpali-v1.2"
DEVICE = "cuda"                         # 或 "mps" / "cpu"
VIDORE_SUBSETS = [                      # ViDoRe 评估数据集
    "vidore/docvqa_test_subsampled",
    "vidore/infovqa_test_subsampled",
]
```

### 2. 运行基准评估（干净图片）

```bash
python experiments/run_benchmark.py --condition clean
# 结果保存到 results/results_clean.json
```

### 3. 运行退化评估

```bash
# 可选退化类型：light_noise, heavy_noise, motion_blur, tilt, jpeg_low, combined
python experiments/run_benchmark.py --condition degraded --deg heavy_noise
python experiments/run_benchmark.py --condition degraded --deg motion_blur
python experiments/run_benchmark.py --condition degraded --deg tilt
python experiments/run_benchmark.py --condition degraded --deg jpeg_low
```

### 4. 运行复原评估

```bash
# 可选复原方法：nlmeans, gaussian, wiener
python experiments/run_benchmark.py --condition restored --deg heavy_noise --rest nlmeans
python experiments/run_benchmark.py --condition restored --deg heavy_noise --rest wiener
```

### 5. 运行文档分割评估（同学B）

```bash
python experiments/run_benchmark.py --condition segmented
```

### 6. PSO 参数寻优（同学D）

```bash
# 自动搜索最优的 NLMeans 参数，最大化 nDCG@5
python experiments/pso_optimize.py
# 结果保存到 results/results_pso.json
```

### 7. 生成对比图表

```bash
python experiments/visualize_results.py
# 生成 results/comparison_chart.png
```

---

## 完整实验流程

```bash
# === 第一步：基准评估 ===
python experiments/run_benchmark.py --condition clean

# === 第二步：各类退化 ===
for DEG in light_noise heavy_noise motion_blur tilt jpeg_low combined; do
    python experiments/run_benchmark.py --condition degraded --deg $DEG
done

# === 第三步：复原策略 ===
for REST in nlmeans wiener gaussian; do
    python experiments/run_benchmark.py --condition restored --deg heavy_noise --rest $REST
done

# === 第四步：文档分割 ===
python experiments/run_benchmark.py --condition segmented

# === 第五步：PSO 寻优 ===
python experiments/pso_optimize.py

# === 第六步：生成图表 ===
python experiments/visualize_results.py
```

预期输出的 `results/comparison_chart.png` 会展示如下对比：

```
nDCG@5
1.0 ┤  ████
0.8 ┤  ████  ████
0.6 ┤  ████  ████  ████         ████  ████
0.4 ┤  ████  ████  ████  ████  ████  ████  ████
    └──────────────────────────────────────────
      清洁   轻噪  重噪  模糊  倾斜  NLM   Wiener 分割
      基准   ←──────退化──────→  ←──复原──→
```

---

## 模块 API 说明

### 退化 Pipeline

```python
from robust.degradation.pipeline import DegradationPipeline

# 创建退化组合
pipeline = DegradationPipeline([
    ("gaussian_noise", {"std": 30}),        # 高斯噪声，标准差30
    ("motion_blur",    {"kernel_size": 15, "angle": 45}),  # 运动模糊
])

from PIL import Image
img = Image.open("document.png")
degraded = pipeline(img)   # 返回 PIL.Image
```

**支持的退化类型：**

| 类型名 | 参数 | 说明 |
|--------|------|------|
| `gaussian_noise` | `std` (默认25) | 高斯白噪声 |
| `salt_pepper_noise` | `amount` (默认0.05) | 椒盐噪声 |
| `gaussian_blur` | `sigma` (默认3) | 高斯模糊 |
| `motion_blur` | `kernel_size`, `angle` | 运动模糊 |
| `tilt` | `angle` (默认10) | 文档倾斜（度） |
| `jpeg_compression` | `quality` (默认10) | JPEG 压缩质量 1-95 |
| `watermark` | `text`, `alpha` | 文字水印 |

### 复原 Pipeline

```python
from robust.restoration.pipeline import RestorationPipeline

pipeline = RestorationPipeline([
    ("nlmeans",  {"h": 10}),          # NLMeans 去噪
    ("gaussian", {"sigma": 1.5}),     # 轻微高斯平滑
])
restored = pipeline(degraded_img)
```

**支持的复原方法：**

| 方法名 | 参数 | 说明 |
|--------|------|------|
| `nlmeans` | `h` (默认10) | OpenCV Non-Local Means，最佳质量 |
| `gaussian` | `sigma` (默认1.5) | 高斯平滑，速度最快 |
| `wiener` | `noise_power` (默认0.01) | Wiener 滤波，频域去模糊 |

### 文档分割

```python
from robust.segmentation.document_seg import segment_document

segmented = segment_document(img, padding=5)
# 返回背景置白的文档图像，减少 ColQwen2 的冗余 patch
```

### 评估指标

```python
from robust.evaluation.metrics import ndcg_at_k, recall_at_k, mean_reciprocal_rank

scores  = [0.9, 0.3, 0.7, 0.1]   # 4 个文档的检索得分
relevant = {0}                     # 第 0 号是正确答案

ndcg  = ndcg_at_k(scores, relevant, k=5)
recall = recall_at_k(
    sorted(range(4), key=lambda i: scores[i], reverse=True),
    relevant, k=5
)
mrr = mean_reciprocal_rank(
    sorted(range(4), key=lambda i: scores[i], reverse=True),
    relevant
)
```

### PSO 参数寻优

```python
from robust.optimization.pso import PSOptimizer

optimizer = PSOptimizer(
    bounds=[(1, 30), (0.1, 5)],   # [nlmeans_h 范围, gaussian_sigma 范围]
    n_particles=10,
    iters=20,
)
best_params, best_score = optimizer.optimize(my_objective_fn)
# best_params: array([nlmeans_h, sigma])
# best_score: 最优 nDCG@5
```

---

## 结果文件说明

所有实验结果保存在 `results/` 目录下：

| 文件名 | 内容 |
|--------|------|
| `results_clean.json` | 干净图片基准 nDCG@5 |
| `results_degraded_<类型>.json` | 各退化类型的检索性能 |
| `results_restored_<退化>_<复原>.json` | 复原后的检索性能 |
| `results_segmented.json` | 文档分割后的检索性能 |
| `results_pso.json` | PSO 寻优结果（最优参数） |
| `comparison_chart.png` | 所有条件对比图 |

每个 JSON 文件格式：
```json
{
  "docvqa_test_subsampled": {
    "ndcg@5": 0.8234,
    "recall@5": 0.7891,
    "mrr": 0.8012,
    "n_samples": 500
  }
}
```

---

## 技术细节

### 为什么选 ColQwen2 而不是原始 ColPali？

ColQwen2（Qwen2-VL-2B 底座）相比原版 ColPali（PaliGemma-3B）在 ViDoRe 基准上性能提升约 5 个百分点，且显存占用相近，更适合作为研究的高性能基准线。

### 退化参数推荐

| 场景 | 推荐退化配置 |
|------|-------------|
| 手机拍摄文件 | `motion_blur` + `gaussian_noise(std=20)` |
| 低质量扫描仪 | `heavy_noise` + `jpeg_compression(quality=15)` |
| 扫描歪斜 | `tilt(angle=10-15)` |
| 带版权水印 | `watermark(alpha=0.4)` |

### 关于 PSO 寻优

PSO（粒子群优化）在这里用于自动搜索最优的图像复原参数，避免手动调参：
- **搜索空间**：NLMeans 强度 h ∈ [1, 30]，高斯 sigma ∈ [0.1, 5]
- **目标函数**：验证集上的 nDCG@5
- **推荐配置**：`n_particles=10, iters=20`（约 30-60 分钟 GPU 时间）

---

## 注意事项

1. **不要微调模型**：本项目的所有实验均使用 **Inference（推理模式）**，不涉及任何训练，避免算力消耗。

2. **数据集加载**：ViDoRe 数据集会从 HuggingFace Hub 自动下载，首次运行需要网络连接。可通过设置环境变量 `HF_ENDPOINT=https://hf-mirror.com` 使用国内镜像。

3. **OOD 测试（同学C）**：将域外文档图片放入 `data/ood/<类别>/` 目录，并提供 `queries.txt`，然后用 `--subsets` 参数指向本地路径运行评估。

---

## 引用

本项目基于以下工作：

```bibtex
@article{faysse2024colpali,
  title={ColPali: Efficient Document Retrieval with Vision Language Models},
  author={Faysse, Manuel and Sibille, Hugues and Wu, Tony and Omrani, Bilel and
          Viaud, Gautier and Hudelot, Céline and Colombo, Pierre},
  journal={arXiv preprint arXiv:2407.01449},
  year={2024}
}
```

---

## 许可证

本研究代码遵循与原仓库相同的 Apache 2.0 许可证。
