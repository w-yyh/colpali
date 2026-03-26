# ColQwen2-v1.0 环境安装与测试指南

## 前提条件

- Linux 系统（已在 Ubuntu 上验证）
- NVIDIA GPU（显存 ≥ 8GB，已在 RTX 2080 Ti 11GB 上验证）
- Conda 已安装
- 项目中已包含 `colqwen2-v1.0/` 文件夹（LoRA adapter 权重）

> **注意**：首次运行时需要联网下载基础模型 `vidore/colqwen2-base`（约 8.3GB），之后会缓存到 `~/.cache/huggingface/hub/`，后续无需再下载。  
> ViDoRe 数据集同样从 HuggingFace 自动下载到 `~/.cache/huggingface/datasets/`，不占用项目目录。

---

## 安装步骤

### 1. 创建 Conda 环境

```bash
conda create -n colqwen2-test python=3.10 -y
conda activate colqwen2-test
```

### 2. 安装项目依赖

在项目根目录下执行：

```bash
cd /path/to/colpali-segmentation
pip install -e "."
```

### 3. 安装额外研究依赖

```bash
pip install opencv-python-headless matplotlib pytest datasets
```

### 4. 设置 HuggingFace 镜像（国内网络必需）

由于 HuggingFace 官方域名在国内无法直接访问，需要设置镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

建议将此行加入 `~/.bashrc` 以持久生效：

```bash
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
source ~/.bashrc
```

---

## 测试验证

### 运行单元测试（不需要 GPU）

```bash
conda activate colqwen2-test
python -m pytest tests/test_segmentation.py -v
# 预期: 6 passed
```

### 运行模型加载测试

```bash
conda activate colqwen2-test
export HF_ENDPOINT=https://hf-mirror.com
cd /path/to/colpali-segmentation
python tools/test_colqwen2.py
```

看到 "测试通过" 即表示环境安装正确、模型可正常加载和推理。

---

## 运行实验

### 分割实验（完整流程）

```bash
conda activate colqwen2-test
export HF_ENDPOINT=https://hf-mirror.com

# 前台运行（可指定 GPU 设备和模型路径）
python experiments/run_segmentation_experiment.py --device cuda:0

# 使用本地模型（如果网络不稳定）
python experiments/run_segmentation_experiment.py --device cuda:0 --model ./colqwen2-v1.0
```

### 后台运行（推荐，适合长时间实验）

```bash
# nohup 方式，关闭终端/SSH 也不会停
nohup env HF_ENDPOINT=https://hf-mirror.com \
    python experiments/run_segmentation_experiment.py \
    --device cuda:0 --model ./colqwen2-v1.0 \
    > outputs/experiment_console.log 2>&1 &

# 查看进度
tail -f outputs/experiment_console.log
```

### 通过通用基准框架运行

```bash
# 运行 clean baseline
python experiments/run_benchmark.py --condition clean

# 运行分割实验
python experiments/run_benchmark.py --condition segmented
```

### 实验输出

每次运行会在 `outputs/<YYYYMMDD_HHMMSS>/` 下生成：

```
outputs/20260326_140641/
├── all_results.json           # 所有条件的完整指标
├── experiment_log.json        # 实验元信息
├── summary.txt                # 文本格式的结果摘要
├── comparison_chart.png       # nDCG@5 对比柱状图
└── visualizations/            # 分割前后对比图
```

---

## 数据集说明

ViDoRe 数据集**自动从 HuggingFace 下载**到默认缓存目录 `~/.cache/huggingface/datasets/`，不需要手动下载，也不会占用项目 `data/` 目录。

首次下载数据集大小：
- `vidore/docvqa_test_subsampled`: ~292MB, 500 样本
- `vidore/infovqa_test_subsampled`: ~219MB, 500 样本

---

## 常见问题

### Q: 报错 `OSError: Can't load the configuration of 'vidore/colqwen2-base'`

基础模型未缓存且网络不通。请确保设置了 `HF_ENDPOINT=https://hf-mirror.com` 并保持网络畅通，首次运行需下载约 8.3GB 的基础模型。

### Q: 报错 `CUDA out of memory`

模型需要约 5-6GB 显存。确保 GPU 有足够空闲显存，或通过 `nvidia-smi` 检查。也可以修改 `experiments/config.py` 中 `BATCH_SIZE = 2` 降低显存占用。

### Q: 想使用特定 GPU 设备

```bash
python experiments/run_segmentation_experiment.py --device cuda:1
```

### Q: nohup 运行时报 HuggingFace 网络错误

nohup 子进程不继承当前 shell 的环境变量，需要在命令中显式传入：
```bash
nohup env HF_ENDPOINT=https://hf-mirror.com python ... > log.txt 2>&1 &
```
