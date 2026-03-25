# ColQwen2-v1.0 环境安装与测试指南

## 前提条件

- Linux 系统（已在 Ubuntu 上验证）
- NVIDIA GPU（显存 ≥ 8GB，已在 RTX 2080 Ti 11GB 上验证）
- Conda 已安装
- 项目中已包含 `colqwen2-v1.0/` 文件夹（LoRA adapter 权重）

> **注意**：首次运行时需要联网下载基础模型 `vidore/colqwen2-base`（约 8.3GB），之后会缓存到 `~/.cache/huggingface/hub/`，后续无需再下载。

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

这会自动安装以下核心依赖：

| 包 | 版本 |
|---|------|
| torch | 2.10.0 |
| transformers | 5.3.0 |
| peft | 0.18.1 |
| accelerate | 1.13.0 |
| colpali_engine | 本地（可编辑模式） |

### 3. 设置 HuggingFace 镜像（国内网络必需）

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

### 运行测试脚本

```bash
conda activate colqwen2-test
export HF_ENDPOINT=https://hf-mirror.com
cd /path/to/colpali-segmentation
python document/test_colqwen2.py
```

### 预期输出

```
============================================================
ColQwen2-v1.0 本地模型测试
============================================================

[1/4] 加载模型...
  模型加载成功，设备: cuda:0

[2/4] 加载处理器...
  处理器加载成功

[3/4] 准备测试数据...
  测试图片数: 2, 测试查询数: 2

[4/4] 执行推理...
  相似度分数矩阵:
tensor([[2.6406, 2.5000],
        [6.3438, 5.8438]])

============================================================
测试通过！ColQwen2-v1.0 本地模型加载与推理正常。
============================================================
```

看到 "测试通过" 即表示环境安装正确、模型可正常加载和推理。

---

## 测试脚本说明

[test_colqwen2.py](test_colqwen2.py) 做了以下事情：

1. 从本地 `colqwen2-v1.0/` 加载 ColQwen2 模型（bfloat16，CUDA）
2. 加载对应的处理器（tokenizer + image processor）
3. 用两张纯色测试图片和两条英文查询进行推理
4. 计算 query-image 的 multi-vector 相似度分数矩阵

---

## 常见问题

### Q: 报错 `OSError: Can't load the configuration of 'vidore/colqwen2-base'`

基础模型未缓存且网络不通。请确保设置了 `HF_ENDPOINT=https://hf-mirror.com` 并保持网络畅通，首次运行需下载约 8.3GB 的基础模型。

### Q: 报错 `CUDA out of memory`

模型需要约 5-6GB 显存。确保 GPU 有足够空闲显存，或通过 `nvidia-smi` 检查是否有其他进程占用。

### Q: 想使用 CPU 运行

将测试脚本中的 `device_map="cuda:0"` 改为 `device_map="cpu"`，速度会较慢但可以运行。
