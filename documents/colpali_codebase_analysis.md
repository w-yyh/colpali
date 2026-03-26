# ColPali 代码库完整解析：从论文到代码的训练与推理全流程

> **目标**：结合 ColPali 论文的 Method 部分，逐行解读 `colpali_engine` 仓库的核心代码，覆盖模型架构、损失函数、数据处理、训练流程、推理评分、可解释性和压缩模块。每个模块都会给出**代码位置、张量形状变化、关键函数逐行分析**。

---

## 目录

1. [仓库总览与项目定位](#1-仓库总览与项目定位)
2. [核心概念回顾：论文 Method](#2-核心概念回顾论文-method)
3. [模型架构详解](#3-模型架构详解)
   - 3.1 [ColPali (PaliGemma)](#31-colpali-paligemma)
   - 3.2 [BiPali (单向量基线)](#32-bipali-单向量基线)
   - 3.3 [ColQwen2 (Qwen2-VL)](#33-colqwen2-qwen2-vl)
   - 3.4 [Processor 处理器](#34-processor-处理器)
4. [损失函数与延迟交互](#4-损失函数与延迟交互)
   - 4.1 [ColbertLoss (InfoNCE)](#41-colbertloss-infonce)
   - 4.2 [ColbertPairwiseCELoss](#42-colbertpairwiseceloss)
   - 4.3 [ColbertNegativeCELoss](#43-colbertnegativeceloss)
5. [数据处理管线](#5-数据处理管线)
   - 5.1 [Dataset 与 Corpus](#51-dataset-与-corpus)
   - 5.2 [VisualRetrieverCollator](#52-visualretrievercollator)
   - 5.3 [SingleDatasetBatchSampler](#53-singledatasetbatchsampler)
6. [训练流程](#6-训练流程)
   - 6.1 [ColModelTraining 配置与启动](#61-colmodeltraining-配置与启动)
   - 6.2 [ContrastiveTrainer 核心训练逻辑](#62-contrastivetrainer-核心训练逻辑)
   - 6.3 [ColModelTorchTraining 分布式训练](#63-colmodeltorchtraining-分布式训练)
7. [推理与评分](#7-推理与评分)
   - 7.1 [MaxSim 评分函数](#71-maxsim-评分函数)
   - 7.2 [FastPlaid 索引加速](#72-fastplaid-索引加速)
8. [可解释性模块](#8-可解释性模块)
9. [Token 压缩模块](#9-token-压缩模块)
10. [ViDoRe 基准复现指南](#10-vidore-基准复现指南)

---

## 1. 仓库总览与项目定位

### 1.1 这个仓库是做什么的？

`colpali_engine` 是 ColPali 论文的官方代码实现，它的核心目标是：

> **用视觉语言模型（VLM）直接对文档页面图像进行嵌入，跳过传统的 OCR/PDF解析/分块流水线，实现端到端的文档检索。**

传统文档检索流程：
```
PDF → OCR提取文本 → 布局检测 → 分块 → 文本嵌入 → 检索
```

ColPali 的流程：
```
PDF页面截图 → VLM直接编码为多向量嵌入 → 延迟交互(MaxSim)检索
```

### 1.2 仓库结构

```
colpali_engine/
├── models/           # 所有模型架构（ColPali, ColQwen2, BiPali 等）
│   ├── paligemma/    # 基于 PaliGemma-3B 的模型
│   ├── qwen2/        # 基于 Qwen2-VL 的模型
│   ├── idefics3/     # 基于 Idefics3 的模型
│   ├── gemma3/       # 基于 Gemma3 的模型
│   ├── qwen3/        # 基于 Qwen3 的模型
│   └── ...           # 更多后续模型
├── loss/             # 损失函数（延迟交互 + 双编码器）
│   ├── late_interaction_losses.py   # ColBERT风格损失
│   └── bi_encoder_losses.py         # 双编码器损失
├── collators/        # 数据整理器
│   └── visual_retriever_collator.py
├── data/             # 数据集和采样器
│   ├── dataset.py
│   └── sampler.py
├── trainer/          # 训练器
│   ├── colmodel_training.py         # HuggingFace Trainer封装
│   ├── contrastive_trainer.py       # 对比学习训练器
│   └── colmodel_torch_training.py   # 原生PyTorch分布式训练
├── utils/            # 工具函数
│   ├── processing_utils.py          # Processor基类 + MaxSim评分
│   └── torch_utils.py               # 设备管理 + padding工具
├── interpretability/ # 可解释性（相似度热图）
│   ├── similarity_maps.py
│   └── similarity_map_utils.py
└── compression/      # Token压缩（层次聚类池化）
    └── token_pooling/
```

### 1.3 模型家族总览

仓库中每个 VLM backbone 都提供两种变体：

| 变体 | 嵌入类型 | 输出形状 | 检索策略 |
|------|---------|---------|---------|
| **Col*** (如 ColPali, ColQwen2) | 多向量 | `(batch, seq_len, 128)` | 延迟交互 MaxSim |
| **Bi*** (如 BiPali, BiQwen2) | 单向量 | `(batch, hidden_dim)` | 余弦相似度 |

论文已证明：**多向量 + 延迟交互 >> 单向量 + 余弦相似度**。

### 1.4 关键依赖

```toml
# pyproject.toml
torch>=2.2.0          # 深度学习框架
transformers>=5.2.0   # HuggingFace，提供 PaliGemma/Qwen2VL 等模型
peft>=0.18.0          # LoRA 适配器
accelerate            # 分布式训练 (可选)
bitsandbytes          # 8-bit 优化器 (可选)
einops                # 张量重排 (可解释性模块)
scipy                 # 层次聚类 (Token 压缩)
```

---

## 2. 核心概念回顾：论文 Method

### 2.1 架构思想

ColPali 的核心思想是把 **视觉语言模型（VLM）** 适配为 **文档检索器**：

1. **文档端**：将文档页面图像输入 VLM，利用 VLM 的图像编码器将页面切成 patches，经过语言模型得到**每个 patch 的嵌入向量**（多向量表示）
2. **查询端**：将文本查询输入 VLM 的语言模型，得到**每个 token 的嵌入向量**
3. **匹配**：使用 ColBERT 的延迟交互机制（MaxSim），计算查询 token 与文档 patch 之间的最大相似度之和

论文公式（公式 1）：

$$LI(q,d) = \sum_{i=1}^{N_q} \max_{j=1}^{N_d} \langle E_q(i) \mid E_d(j) \rangle$$

其中：
- $E_q \in \mathbb{R}^{N_q \times D}$：查询的多向量表示，$N_q$ 个 token，每个 $D=128$ 维
- $E_d \in \mathbb{R}^{N_d \times D}$：文档的多向量表示，$N_d$ 个 patch，每个 $D=128$ 维
- $\langle \cdot \mid \cdot \rangle$：点积操作

### 2.2 为什么用多向量而非单向量？

文档页面包含大量视觉信息（文本、表格、图表、布局），用单一向量压缩会丢失细节。多向量表示**保留了每个图像 patch 的独立语义**，允许查询中的不同词汇与文档中不同区域精确匹配。

### 2.3 训练损失

论文公式（公式 2）—— 批内对比损失：

$$L = -\frac{1}{b} \sum_{k=1}^{b} \log \frac{\exp(s^+_k)}{\exp(s^+_k) + \exp(s^-_k)}$$

其中 $s^+_k = LI(q_k, d_k)$ 是正样本得分，$s^-_k = \max_{l \neq k} LI(q_k, d_l)$ 是批内最难负样本得分。

### 2.4 查询增强

按照 ColBERT 的做法，在查询 token 后附加若干特殊 token（如 `<unused0>` 或 pad token），作为**可微的查询扩展缓冲区**，让模型学习隐式的查询重写。

---

## 3. 模型架构详解

### 3.1 ColPali (PaliGemma)

> **代码位置**：`colpali_engine/models/paligemma/colpali/modeling_colpali.py`

ColPali 是整个项目的**起源模型**，基于 Google 的 PaliGemma-3B（SigLIP 视觉编码器 + Gemma-2B 语言模型）。

#### 3.1.1 类定义与初始化

```python
# 文件: colpali_engine/models/paligemma/colpali/modeling_colpali.py

class ColPali(PaliGemmaPreTrainedModel):
    """
    ColPali model implementation from the "ColPali: Efficient Document Retrieval
    with Vision Language Models" paper.
    """

    main_input_name: ClassVar[str] = "doc_input_ids"  # transformers 要求的属性

    def __init__(self, config: PaliGemmaConfig, mask_non_image_embeddings: bool = False):
        super().__init__(config=config)

        # 1. 实例化完整的 PaliGemma 生成模型（SigLIP + Gemma）
        model = PaliGemmaForConditionalGeneration(config=config)
        # 处理权重绑定（language model 的 input/output embeddings 共享）
        if model.model.language_model._tied_weights_keys is not None:
            self._tied_weights_keys = [
                f"model.model.language_model.{k}" for k in model.model.language_model._tied_weights_keys
            ]
        self.model = model

        # 2. 投影层：hidden_size → 128
        #    这是 ColBERT 论文中使用的低维嵌入空间
        self.dim = 128
        self.custom_text_proj = nn.Linear(self.model.config.text_config.hidden_size, self.dim)

        # 3. 是否在推理时只保留图像 patch 嵌入
        self.mask_non_image_embeddings = mask_non_image_embeddings

        self.post_init()  # HuggingFace 标准初始化钩子
```

**关键设计解析**：

- **`PaliGemmaForConditionalGeneration`**：这是 HuggingFace Transformers 中已实现的完整 PaliGemma 模型。ColPali 并没有从头写 VLM，而是**复用现有模型然后添加投影层**。这是一个非常高效的工程策略。
- **`self.dim = 128`**：投影到 128 维而非使用原始的 hidden_size（2048），这是 ColBERT 论文的做法，目的是减少存储开销。每个 patch 存一个 128 维向量，比存 2048 维省 16 倍空间。
- **`mask_non_image_embeddings`**：推理时可选只保留图像 token 的嵌入，丢弃文本 prompt 的嵌入（如 "Describe the image." 产生的 token）。论文消融实验表明这不是必须的。

#### 3.1.2 forward() 方法 —— 核心前向传播

```python
# 文件: colpali_engine/models/paligemma/colpali/modeling_colpali.py

def forward(self, *args, **kwargs) -> torch.Tensor:
    # 清理 kwargs，确保 output_hidden_states 由我们控制
    kwargs.pop("output_hidden_states", None)

    # 将 pixel_values 转为模型精度（如 bfloat16）
    if "pixel_values" in kwargs:
        kwargs["pixel_values"] = kwargs["pixel_values"].to(dtype=self.dtype)

    # ===== 步骤1: 通过完整 VLM 前向传播 =====
    outputs = self.model(*args, output_hidden_states=True, **kwargs)
    # outputs.hidden_states 是一个元组，包含每一层的隐藏状态
    # outputs.hidden_states[-1] 就是最后一层的输出

    last_hidden_states = outputs.hidden_states[-1]
    # 张量形状: (batch_size, sequence_length, hidden_size)
    # 例如: (4, 1030, 2048) —— 4张图片，每张1024个patch + 6个文本token

    # ===== 步骤2: 投影到低维空间 =====
    proj = self.custom_text_proj(last_hidden_states)
    # 张量形状: (batch_size, sequence_length, 128)
    # nn.Linear(2048, 128) 将每个 token 的表示压缩到 128 维

    # ===== 步骤3: L2 归一化 =====
    proj = proj / proj.norm(dim=-1, keepdim=True)
    # 归一化后每个向量的L2范数为1
    # 这使得后续的点积操作等价于余弦相似度
    # 张量形状不变: (batch_size, sequence_length, 128)

    # ===== 步骤4: 应用 attention_mask =====
    proj = proj * kwargs["attention_mask"].unsqueeze(-1)
    # attention_mask: (batch_size, sequence_length) → unsqueeze → (batch_size, sequence_length, 1)
    # 将 padding 位置的嵌入清零，这样在后续 MaxSim 计算中它们不会贡献分数
    # 张量形状不变: (batch_size, sequence_length, 128)

    # ===== 步骤5 (可选): 只保留图像 patch 嵌入 =====
    if "pixel_values" in kwargs and self.mask_non_image_embeddings:
        image_mask = (kwargs["input_ids"] == self.config.image_token_index).unsqueeze(-1)
        # image_mask: (batch_size, sequence_length, 1)，只有图像token位置为True
        proj = proj * image_mask  # 文本token的嵌入被清零

    return proj  # 最终输出: (batch_size, sequence_length, 128)
```

**张量流动全图**：

```
输入图像 (224×224 RGB)
    ↓ SigLIP Vision Encoder (patch_size=14)
    ↓ → 16×16 = 256 个 patch（或更多，取决于分辨率）
    ↓ Multi-Modal Projector
    ↓ 与文本 token 拼接
    ↓
VLM 语言模型 (Gemma-2B)
    ↓ 所有层的 Transformer 计算
    ↓
hidden_states[-1]: (batch, seq_len, 2048)
    ↓ nn.Linear(2048, 128)
    ↓
proj: (batch, seq_len, 128)
    ↓ L2 normalize
    ↓ mask padding
    ↓
output: (batch, seq_len, 128)  — 多向量嵌入
```

#### 3.1.3 为什么取 `hidden_states[-1]` 而不是用 `logits`？

因为 ColPali 不做生成任务，不需要预测下一个 token。它需要的是**每个位置的上下文化表示**（即最后一层隐藏状态），然后投影到检索用的低维空间。`logits` 是经过 `lm_head` 映射到词汇表大小的输出，对于检索任务毫无意义。

#### 3.1.4 patch_size 属性

```python
@property
def patch_size(self) -> int:
    return self.model.vision_tower.config.patch_size
```

这个属性返回视觉编码器的 patch 大小（通常为 14），用于计算图像被切分成多少个 patch：`n_patches = image_size // patch_size`。

---

### 3.2 BiPali (单向量基线)

> **代码位置**：`colpali_engine/models/paligemma/bipali/modeling_bipali.py`

BiPali 是论文中的**消融基线**，证明多向量表示比单向量更好。

```python
# 文件: colpali_engine/models/paligemma/bipali/modeling_bipali.py

class BiPali(PaliGemmaPreTrainedModel):
    """
    BiPali: 表示被平均池化为单一向量。
    """

    def forward(self, *args, **kwargs):
        kwargs.pop("output_hidden_states", None)
        if "pixel_values" in kwargs:
            kwargs["pixel_values"] = kwargs["pixel_values"].to(dtype=self.dtype)

        outputs = self.model(*args, output_hidden_states=True, **kwargs)
        last_hidden_states = outputs.hidden_states[-1]
        # (batch_size, sequence_length, hidden_size)

        # ===== 关键区别：平均池化 =====
        proj = torch.sum(
            last_hidden_states * kwargs["attention_mask"].unsqueeze(-1), dim=1
        ) / torch.sum(
            kwargs["attention_mask"], dim=1, keepdim=True
        )
        # 分子: 将 padding 位置清零后沿 seq_len 维度求和
        #   → (batch_size, hidden_size)
        # 分母: 有效 token 数量
        #   → (batch_size, 1)
        # 结果: (batch_size, hidden_size) — 每个样本一个向量

        proj = proj / proj.norm(dim=-1, keepdim=True)  # L2 归一化
        return proj  # (batch_size, hidden_size)
```

**ColPali vs BiPali 的核心区别**：

| 特性 | ColPali | BiPali |
|------|---------|--------|
| 输出维度 | `(batch, seq_len, 128)` | `(batch, 2048)` |
| 每文档向量数 | ~1030 个向量 | 1 个向量 |
| 匹配方式 | MaxSim (延迟交互) | 点积/余弦 |
| 性能 (ViDoRe nDCG@5) | **81.3** | 58.8 |

论文结论：多向量表示能保留文档页面中的空间信息，使得检索性能提升 22.5 个百分点。

---

### 3.3 ColQwen2 (Qwen2-VL)

> **代码位置**：`colpali_engine/models/qwen2/colqwen2/modeling_colqwen2.py`

ColQwen2 是论文消融实验中的"更好的 VLM = 更好的检索器"的证明，比 ColPali 高 5.3 nDCG@5。

```python
# 文件: colpali_engine/models/qwen2/colqwen2/modeling_colqwen2.py

class ColQwen2(Qwen2VLModel):
    """
    ColQwen2: 基于 Qwen2-VL 的 ColBERT 风格文档检索模型
    """

    def __init__(self, config: Qwen2VLConfig, mask_non_image_embeddings: bool = False):
        super().__init__(config=config)
        # 动态获取 hidden_size（兼容不同配置格式）
        hidden_size = getattr(self.config, "hidden_size", None)
        if hidden_size is None and hasattr(self.config, "text_config"):
            hidden_size = getattr(self.config.text_config, "hidden_size", None)

        self.dim = 128
        self.custom_text_proj = nn.Linear(hidden_size, self.dim)
        self.padding_side = "left"  # Qwen2 使用左填充
        self.mask_non_image_embeddings = mask_non_image_embeddings
        self.post_init()
```

#### 3.3.1 forward() —— Qwen2 特有的 pixel_values 处理

```python
# 文件: colpali_engine/models/qwen2/colqwen2/modeling_colqwen2.py

def forward(self, *args, **kwargs) -> torch.Tensor:
    # ===== Qwen2 特有：处理 padded pixel_values =====
    if "pixel_values" in kwargs:
        offsets = kwargs["image_grid_thw"][:, 1] * kwargs["image_grid_thw"][:, 2]
        # image_grid_thw: (batch_size, 3) → [temporal, height_patches, width_patches]
        # offsets: (batch_size,) → 每张图片的实际 patch 数量
        #
        # 为什么需要这一步？因为 ColQwen2Processor 为了兼容 DDP，
        # 把不同长度的 pixel_values 做了 pad_sequence 对齐。
        # 这里需要把 padding 去掉，恢复成原始的 concatenated 格式。
        kwargs["pixel_values"] = torch.cat(
            [pixel_sequence[:offset]
             for pixel_sequence, offset in zip(kwargs["pixel_values"], offsets)],
            dim=0,
        )
        # 从 (batch_size, max_patches, pixel_dim) 变成 (total_patches, pixel_dim)
        # 这是 Qwen2VLModel 期望的输入格式

    # 清理不需要的参数
    kwargs.pop("return_dict", True)
    kwargs.pop("output_hidden_states", None)
    kwargs.pop("use_cache", None)

    # ===== 通过 Qwen2VL 前向传播 =====
    hidden_states = (
        super()
        .forward(*args, **kwargs, use_cache=False, output_hidden_states=True, return_dict=True)
        .last_hidden_state
    )  # (batch_size, sequence_length, hidden_size)

    # ===== 投影 + 归一化（与 ColPali 相同） =====
    proj = self.custom_text_proj(hidden_states)  # (batch_size, seq_len, 128)
    proj = proj / proj.norm(dim=-1, keepdim=True)  # L2 归一化
    proj = proj * kwargs["attention_mask"].unsqueeze(-1)  # mask padding

    if "pixel_values" in kwargs and self.mask_non_image_embeddings:
        image_mask = (kwargs["input_ids"] == self.config.image_token_id).unsqueeze(-1)
        proj = proj * image_mask

    return proj  # (batch_size, sequence_length, 128)
```

**ColQwen2 vs ColPali 的架构区别**：

| 特性 | ColPali | ColQwen2 |
|------|---------|----------|
| 视觉编码器 | SigLIP-So400m | Qwen2-VL ViT |
| 语言模型 | Gemma-2B | Qwen2-1.5B/7B |
| 动态分辨率 | 否（固定分辨率） | 是（`smart_resize`） |
| padding 方向 | 右填充 | **左填充** |
| pixel_values 格式 | `(batch, channels, H, W)` | `(total_patches, patch_dim)` packed |
| 父类 | `PaliGemmaPreTrainedModel` | `Qwen2VLModel` |

---

### 3.4 Processor 处理器

Processor 负责将原始输入（PIL 图像、文本字符串）转换为模型可以消费的张量。

#### 3.4.1 ColPaliProcessor

> **代码位置**：`colpali_engine/models/paligemma/colpali/processing_colpali.py`

```python
# 文件: colpali_engine/models/paligemma/colpali/processing_colpali.py

class ColPaliProcessor(BaseVisualRetrieverProcessor, PaliGemmaProcessor):
    """
    双重继承：
    - BaseVisualRetrieverProcessor: 提供 score_multi_vector 等评分方法
    - PaliGemmaProcessor: 提供图像/文本处理的底层实现
    """

    # 视觉提示前缀 —— 在每张图片前加上这段文本
    visual_prompt_prefix: ClassVar[str] = "<image><bos>Describe the image."
```

**`process_images()` —— 处理文档页面图像**：

```python
def process_images(self, images: List[Image.Image]) -> Union[BatchFeature, BatchEncoding]:
    images = [image.convert("RGB") for image in images]  # 确保 RGB 格式

    batch_doc = self(
        text=[self.visual_prompt_prefix] * len(images),  # 每张图配 "Describe the image."
        images=images,
        return_tensors="pt",
        padding="longest",  # pad 到批次中最长的序列
    )
    return batch_doc
```

**为什么要加 `"<image><bos>Describe the image."` 前缀？**

这是 PaliGemma 模型的标准输入格式。`<image>` 标记告诉模型这里要插入图像 patch tokens，`<bos>` 是序列开始标记，`"Describe the image."` 是一个简短的任务指令。这些文本 token 会和图像 patch tokens 一起经过语言模型处理，最终所有位置（包括文本和图像）都产生嵌入。

**`process_texts()` —— 处理查询文本**：

```python
def process_texts(self, texts: List[str]) -> Union[BatchFeature, BatchEncoding]:
    return self.tokenizer(
        [self.tokenizer.bos_token + text for text in texts],  # 加 BOS 前缀
        text_pair=None,
        return_token_type_ids=False,
        return_tensors="pt",
        padding="longest",
    )
```

**`score()` —— 调用 MaxSim 评分**：

```python
def score(self, qs, ps, device=None, **kwargs) -> torch.Tensor:
    return self.score_multi_vector(qs, ps, device=device, **kwargs)
```

#### 3.4.2 ColQwen2Processor

> **代码位置**：`colpali_engine/models/qwen2/colqwen2/processing_colqwen2.py`

```python
# 文件: colpali_engine/models/qwen2/colqwen2/processing_colqwen2.py

class ColQwen2Processor(BaseVisualRetrieverProcessor, Qwen2VLProcessor):

    visual_prompt_prefix: ClassVar[str] = (
        "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
        "Describe the image.<|im_end|><|endoftext|>"
    )
    query_augmentation_token: ClassVar[str] = "<|endoftext|>"
    # ColPali 用 pad_token 做查询增强，ColQwen2 用 <|endoftext|>

    def __init__(self, ...):
        super().__init__(...)
        self.tokenizer.padding_side = "left"  # Qwen2 系列统一用左填充
```

**`process_images()` —— Qwen2 的特殊 pixel_values 处理**：

```python
def process_images(self, images: List[Image.Image]) -> Union[BatchFeature, BatchEncoding]:
    images = [image.convert("RGB") for image in images]

    batch_doc = self(
        text=[self.visual_prompt_prefix] * len(images),
        images=images,
        padding="longest",
        return_tensors="pt",
    )

    # ===== Qwen2 特有：pad pixel_values 以兼容 DDP =====
    offsets = batch_doc["image_grid_thw"][:, 1] * batch_doc["image_grid_thw"][:, 2]
    # offsets: 每张图的实际 patch 数

    # 将 concatenated pixel_values 按图片切分
    pixel_values = list(torch.split(batch_doc["pixel_values"], offsets.tolist()))
    # [(patches_img0, dim), (patches_img1, dim), ...]

    # pad 到相同长度，这样在 DDP 多卡训练时形状一致
    batch_doc["pixel_values"] = torch.nn.utils.rnn.pad_sequence(
        pixel_values, batch_first=True
    )
    # (batch_size, max_patches, dim)

    return batch_doc
```

**为什么 Qwen2 需要做 pad_sequence？**

Qwen2-VL 支持动态分辨率——不同图片被切成不同数量的 patch（通过 `smart_resize` 计算）。在单卡推理时，Qwen2VLModel 直接处理 concatenated 的 pixel_values；但在 DDP 多卡训练时，每张卡的 tensor 形状必须一致才能做 `all_gather`，所以 Processor 端做了 pad，Model 端的 `forward()` 再做 unpad。

**`get_n_patches()` —— 计算 patch 网格大小**：

```python
def get_n_patches(self, image_size: Tuple[int, int], spatial_merge_size: int) -> Tuple[int, int]:
    patch_size = self.image_processor.patch_size

    # smart_resize: Qwen2VL 的动态分辨率调整函数
    height_new, width_new = smart_resize(
        width=image_size[0],
        height=image_size[1],
        factor=patch_size * self.image_processor.merge_size,
        min_pixels=self.image_processor.size["shortest_edge"],
        max_pixels=self.image_processor.size["longest_edge"],
    )

    n_patches_x = width_new // patch_size // spatial_merge_size
    n_patches_y = height_new // patch_size // spatial_merge_size

    return n_patches_x, n_patches_y
```

这个函数在可解释性模块中很重要——它告诉我们图像被切成了多少行多少列的 patch，用于将 1D 的 patch 嵌入序列重排为 2D 的空间网格。

---

## 4. 损失函数与延迟交互

> **代码位置**：`colpali_engine/loss/late_interaction_losses.py`

这是整个项目中**数学含量最高**的模块。所有损失函数都继承自 `ColbertModule` 基类。

### 4.1 ColbertModule 基类

```python
# 文件: colpali_engine/loss/late_interaction_losses.py

class ColbertModule(torch.nn.Module):
    """
    所有 ColBERT 损失的基类，封装了共享的工具方法。
    """

    def __init__(
        self,
        max_batch_size: int = 1024,
        tau: float = 0.1,           # smooth-max 的温度
        norm_tol: float = 1e-3,     # 归一化容差
        filter_threshold: float = 0.95,  # 负样本过滤阈值
        filter_factor: float = 0.5,      # 过滤时的衰减因子
    ):
        super().__init__()
        # 预分配索引缓冲区，避免每次前向都创建新张量
        self.register_buffer("idx_buffer", torch.arange(max_batch_size), persistent=False)
```

#### 4.1.1 `_get_idx()` —— 获取正样本索引

```python
def _get_idx(self, batch_size: int, offset: int, device: torch.device):
    idx = self.idx_buffer[:batch_size].to(device)  # [0, 1, 2, ..., batch_size-1]
    return idx, idx + offset  # pos_idx 可能有偏移（多GPU时）
```

**为什么需要 offset？** 在多 GPU 分布式训练中，文档嵌入会通过 `all_gather` 收集所有 GPU 的结果。假设 GPU-0 有 batch_size=32 的文档，GPU-1 也有 32 个，gather 后共 64 个文档。GPU-0 上的查询对应的正样本在 gather 后的索引是 `[0,1,...,31]`，GPU-1 上的查询对应的正样本索引是 `[32,33,...,63]`。所以 GPU-1 需要 offset=32。

#### 4.1.2 `_aggregate()` —— 从 token 级别聚合到文档级别

```python
def _aggregate(
    self,
    scores_raw: torch.Tensor,
    use_smooth_max: bool,
    dim_max: int,
    dim_sum: int,
) -> torch.Tensor:
    if use_smooth_max:
        return self._smooth_max(scores_raw, dim=dim_max).sum(dim=dim_sum)
    return scores_raw.amax(dim=dim_max).sum(dim=dim_sum)
```

这个函数实现了 ColBERT 的核心操作：**先 max 再 sum**。

论文公式：$LI(q,d) = \sum_i \max_j \langle E_q(i) \mid E_d(j) \rangle$

- `dim_max`：在文档 token 维度取 max（对于每个查询 token，找到最匹配的文档 token）
- `dim_sum`：在查询 token 维度求和（把所有查询 token 的最大匹配分数加起来）

`use_smooth_max` 选项用 log-sum-exp 替代 hard max，使梯度更平滑：

```python
def _smooth_max(self, scores: torch.Tensor, dim: int) -> torch.Tensor:
    return self.tau * torch.logsumexp(scores / self.tau, dim=dim)
    # 当 tau → 0 时，logsumexp → max
    # tau 越大，考虑的匹配范围越广（更 "soft"）
```

#### 4.1.3 `_filter_high_negatives()` —— 正样本感知的负样本过滤

```python
def _filter_high_negatives(self, scores: torch.Tensor, pos_idx: torch.Tensor) -> None:
    batch_size = scores.size(0)
    idx = self.idx_buffer[:batch_size].to(scores.device)
    pos_scores = scores[idx, pos_idx]
    # 取出每个查询对应的正样本得分

    thresh = self.filter_threshold * pos_scores.unsqueeze(1)
    # 阈值 = 0.95 × 正样本得分

    mask = scores > thresh
    # 找到得分超过阈值的负样本（这些可能是 "假负样本"——实际上相关的文档被当作负样本）

    mask[idx, pos_idx] = False  # 不要把正样本也过滤了

    scores[mask] *= self.filter_factor
    # 将这些可疑负样本的得分降低（乘以 0.5），减弱它们对损失的贡献
```

**为什么需要这个？** 在批内对比学习中，一个批次中的其他文档被当作负样本。但有时两个不同的查询可能对应相似的文档。如果把实际相关的文档当负样本，会产生错误的梯度信号。这个过滤机制检测并降权这些情况。

---

### 4.2 ColbertLoss (InfoNCE)

这是论文中使用的**主要损失函数**，对应论文公式 (2)。

```python
# 文件: colpali_engine/loss/late_interaction_losses.py

class ColbertLoss(ColbertModule):
    """
    InfoNCE loss for late interaction (ColBERT).
    对应论文公式 (2) 的 Softmax 交叉熵版本。
    """

    def __init__(
        self,
        temperature: float = 0.02,       # 温度参数，控制分布的锐利程度
        normalize_scores: bool = True,    # 是否按查询长度归一化
        use_smooth_max: bool = False,     # 是否用 logsumexp 替代 max
        pos_aware_negative_filtering: bool = False,
        max_batch_size: int = 1024,
        ...
    ):
        ...
        self.ce_loss = CrossEntropyLoss()  # PyTorch 内置交叉熵
```

#### 4.2.1 forward() 逐行解析

```python
def forward(
    self,
    query_embeddings: torch.Tensor,   # (batch_size, query_length, dim=128)
    doc_embeddings: torch.Tensor,     # (batch_size, doc_length, dim=128)
    offset: int = 0                    # 多GPU偏移
) -> torch.Tensor:

    # ===== 步骤1: 计算有效查询长度 =====
    lengths = (query_embeddings[:, :, 0] != 0).sum(dim=1)
    # 利用第0个特征维度是否为0来判断是否是 padding
    # 因为 L2 归一化后的非padding向量，第0维几乎不可能恰好为0
    # lengths: (batch_size,) → 每个查询的有效 token 数

    # ===== 步骤2: 计算所有 query token 和 doc token 的点积 =====
    raw = torch.einsum("bnd,csd->bcns", query_embeddings, doc_embeddings)
    #
    # einsum 解析:
    #   b: 查询的 batch 维度
    #   n: 查询的 token 维度 (query_length)
    #   d: 嵌入维度 (128)
    #   c: 文档的 batch 维度
    #   s: 文档的 token 维度 (doc_length)
    #
    # 输入:
    #   query_embeddings: (B, N_q, D)
    #   doc_embeddings:   (C, N_d, D)  — 多GPU时 C 可能 > B
    #
    # 输出:
    #   raw: (B, C, N_q, N_d)
    #   含义: 第 b 个查询的第 n 个 token 与第 c 个文档的第 s 个 token 的点积
    #
    # 这是整个 ColBERT 计算中最核心、也最占内存的操作！
    # 内存占用: B × C × N_q × N_d × 4 bytes (float32)
    # 例如 B=32, C=32, N_q=30, N_d=1030 ≈ 32×32×30×1030×4 ≈ 122 MB

    # ===== 步骤3: 聚合为文档级别分数 =====
    scores = self._aggregate(raw, self.use_smooth_max, dim_max=3, dim_sum=2)
    # dim_max=3 (N_d): 对每个查询token，在所有文档token中取max
    #   → raw.amax(dim=3): (B, C, N_q)
    # dim_sum=2 (N_q): 将所有查询token的最大匹配分数求和
    #   → .sum(dim=2): (B, C)
    #
    # 这就是论文的 LI(q,d) = Σ_i max_j <E_q(i)|E_d(j)>
    # scores: (B, C) — 每个查询对每个文档的最终匹配分数

    # ===== 步骤4 (可选): 按查询长度归一化 =====
    if self.normalize_scores:
        scores = self._apply_normalization(scores, lengths)
        # scores[b,c] /= lengths[b]
        # 归一化使得不同长度的查询产生的分数可比较

    # ===== 步骤5: 获取正样本索引 =====
    batch_size = scores.size(0)
    idx, pos_idx = self._get_idx(batch_size, offset, scores.device)
    # idx:     [0, 1, 2, ..., B-1]
    # pos_idx: [0+offset, 1+offset, ..., B-1+offset]
    # 含义: 第 b 个查询对应的正样本文档在 doc_embeddings 中的索引

    # ===== 步骤6 (可选): 过滤可疑负样本 =====
    if self.pos_aware_negative_filtering:
        self._filter_high_negatives(scores, pos_idx)

    # ===== 步骤7: 交叉熵损失 =====
    return self.ce_loss(scores / self.temperature, pos_idx)
    # scores / temperature: (B, C) → 缩放后的 logits
    # pos_idx: (B,) → 每个查询对应的正样本文档索引
    #
    # CrossEntropyLoss 实际计算的是:
    #   loss = -log(softmax(scores/T)[b, pos_idx[b]])
    #        = -log(exp(s_pos/T) / Σ_c exp(s_c/T))
    #
    # 当 temperature=0.02 时，分布非常锐利——
    # 正样本的得分需要远远高于所有负样本才能使损失较低。
```

**张量形状变化总结**：

```
query_embeddings: (B, N_q, 128)
doc_embeddings:   (C, N_d, 128)
        ↓ einsum "bnd,csd->bcns"
raw:              (B, C, N_q, N_d)     ← 核心：所有 token 对的点积
        ↓ amax(dim=3)
                  (B, C, N_q)          ← 每个 query token 的最佳匹配
        ↓ sum(dim=2)
scores:           (B, C)              ← 文档级别的匹配分数
        ↓ / temperature
        ↓ CrossEntropyLoss(target=pos_idx)
loss:             scalar               ← 最终损失值
```

---

### 4.3 ColbertPairwiseCELoss

> **代码位置**：`colpali_engine/loss/late_interaction_losses.py`

这是 ColQwen2 训练时实际使用的损失函数（见 `colqwen2-v1.0/training_config.yml`）。

```python
class ColbertPairwiseCELoss(ColbertModule):
    """
    Pairwise softplus loss：不用 softmax 归一化，
    而是直接对比正样本和最难负样本的得分。
    """

    def __init__(self, temperature: float = 1.0, ...):
        # 注意默认温度是 1.0，不是 0.02
        ...

    def forward(self, query_embeddings, doc_embeddings, offset=0):
        lengths = (query_embeddings[:, :, 0] != 0).sum(dim=1)
        raw = torch.einsum("bnd,csd->bcns", query_embeddings, doc_embeddings)
        scores = self._aggregate(raw, self.use_smooth_max, dim_max=3, dim_sum=2)

        if self.normalize_scores:
            scores = self._apply_normalization(scores, lengths)

        batch_size = scores.size(0)
        idx, pos_idx = self._get_idx(batch_size, offset, scores.device)

        if self.pos_aware_negative_filtering:
            self._filter_high_negatives(scores, pos_idx)

        # ===== 关键区别：pairwise softplus =====
        pos_scores = scores.diagonal(offset=offset)
        # 取对角线元素 = 每个查询对应的正样本得分
        # (batch_size,)

        top2 = scores.topk(2, dim=1).values
        # 对每个查询，取得分最高的两个文档
        # top2: (batch_size, 2)

        neg_scores = torch.where(
            top2[:, 0] == pos_scores, top2[:, 1], top2[:, 0]
        )
        # 如果最高分的就是正样本，则取第二高的作为最难负样本
        # 否则取最高的（它就是最难负样本）
        # neg_scores: (batch_size,)

        return F.softplus(
            (neg_scores - pos_scores) / self.temperature
        ).mean()
        # softplus(x) = log(1 + exp(x))
        # 当 neg > pos 时，(neg-pos) > 0，softplus 值大 → 高损失
        # 当 pos >> neg 时，(neg-pos) << 0，softplus ≈ 0 → 低损失
        #
        # 对比 InfoNCE：InfoNCE 考虑所有负样本的得分，
        # 而 Pairwise 只关注最难的一个负样本。
```

**ColbertLoss vs ColbertPairwiseCELoss 对比**：

| 特性 | ColbertLoss | ColbertPairwiseCELoss |
|------|-------------|----------------------|
| 负样本 | 考虑**所有**批内负样本 | 只考虑**最难的一个** |
| 损失函数 | CrossEntropy (softmax) | Softplus (pairwise) |
| 默认温度 | 0.02（很锐利） | 1.0（较平滑） |
| 适用场景 | 基础训练 | 难负样本挖掘 |

---

### 4.4 ColbertNegativeCELoss —— 带显式负样本

```python
# 文件: colpali_engine/loss/late_interaction_losses.py

class ColbertNegativeCELoss(ColbertModule):
    """
    除了批内负样本，还接受额外的显式负样本文档。
    """

    def forward(
        self,
        query_embeddings: torch.Tensor,       # (B, N_q, D)
        doc_embeddings: torch.Tensor,         # (C, N_d, D)
        neg_doc_embeddings: torch.Tensor,     # (B, num_negs, N_neg, D)
        offset: int = 0,
    ) -> torch.Tensor:
        lengths = (query_embeddings[:, :, 0] != 0).sum(dim=1)

        # 正样本：只计算查询与其对应正文档的 token 级交互
        pos_raw = torch.einsum(
            "bnd,bsd->bns",
            query_embeddings,
            doc_embeddings[offset : offset + neg_doc_embeddings.size(0)]
        )
        # einsum "bnd,bsd->bns":
        #   b: batch维度共享 → 查询b 只和 文档b 计算
        #   n: query token, s: doc token, d: embedding dim
        # pos_raw: (B, N_q, N_d)

        # 负样本：查询与每个负文档的 token 级交互
        neg_raw = torch.einsum("bnd,blsd->blns", query_embeddings, neg_doc_embeddings)
        # 多了一个 l 维度 = num_negs（负样本数量）
        # neg_raw: (B, num_negs, N_q, N_neg)

        pos_scores = self._aggregate(pos_raw, ..., dim_max=2, dim_sum=1)
        # (B,) — 每个查询与其正文档的匹配分数

        neg_scores = self._aggregate(neg_raw, ..., dim_max=3, dim_sum=2)
        # (B, num_negs) — 每个查询与每个负文档的匹配分数

        if self.normalize_scores:
            pos_scores = self._apply_normalization(pos_scores, lengths)
            neg_scores = self._apply_normalization(neg_scores, lengths)

        # Softplus: 希望 pos > neg
        loss = F.softplus(
            (neg_scores - pos_scores.unsqueeze(1)) / self.temperature
        ).mean()
        # neg_scores: (B, num_negs)
        # pos_scores.unsqueeze(1): (B, 1)
        # 广播后: (B, num_negs)
        # 对所有 (查询, 负样本) 对计算 softplus

        # 可选：加入批内对比项
        if self.in_batch_term_weight > 0:
            loss_ib = self.inner_loss(query_embeddings, doc_embeddings, offset)
            loss = loss * (1 - self.in_batch_term_weight) + loss_ib * self.in_batch_term_weight

        return loss
```

### 4.5 ColbertSigmoidLoss —— Sigmoid 变体

```python
# 文件: colpali_engine/loss/late_interaction_losses.py

class ColbertSigmoidLoss(ColbertModule):
    def forward(self, query_embeddings, doc_embeddings, offset=0):
        ...
        # 构建标签向量：正样本对为+1，负样本对为-1
        flat_pos = pos_idx * (batch_size + 1)
        pos_mask = -torch.ones(batch_size * batch_size, device=scores.device)
        pos_mask[flat_pos] = 1.0
        # pos_mask[i*B+i] = 1.0（对角线位置）
        # pos_mask[其他] = -1.0

        scores = scores.view(-1) / self.temperature
        # 展平为 (B*B,)

        return F.softplus(-scores * pos_mask).mean()
        # 正样本: softplus(-score * 1) = softplus(-score) → score越大越好
        # 负样本: softplus(-score * -1) = softplus(score) → score越小越好
```

### 4.6 损失函数家族总结

```
ColbertModule (基类)
├── ColbertLoss              — InfoNCE (batch内所有负样本, 交叉熵)
├── ColbertPairwiseCELoss    — Pairwise (只看最难负样本, softplus)
├── ColbertNegativeCELoss    — InfoNCE + 显式负样本
├── ColbertPairwiseNegativeCELoss — Pairwise + 显式负样本
└── ColbertSigmoidLoss       — Sigmoid loss (所有对独立评分)
```

双编码器版本 (`bi_encoder_losses.py`) 结构类似，但使用单向量点积代替延迟交互。

---

## 5. 数据处理管线

### 5.1 Dataset 与 Corpus

> **代码位置**：`colpali_engine/data/dataset.py`

#### 5.1.1 Corpus —— 外部文档库

```python
# 文件: colpali_engine/data/dataset.py

class Corpus:
    """
    管理外部文档语料库，提供通过 ID 检索文档的功能。
    """

    def __init__(
        self,
        corpus_data: List[Dict[str, Any]],            # 文档列表
        docid_to_idx_mapping: Optional[Dict[str, int]] = None,  # ID → 索引映射
        doc_column_name: str = "doc",
    ):
        self.corpus_data = corpus_data
        self.docid_to_idx_mapping = docid_to_idx_mapping
        self.doc_column_name = doc_column_name

    def retrieve(self, docid: Any) -> Document:
        """通过文档 ID 获取文档（图片或文本）"""
        if self.docid_to_idx_mapping is not None:
            doc_idx = self.docid_to_idx_mapping[docid]
        else:
            doc_idx = docid
        return self.corpus_data[doc_idx][self.doc_column_name]
```

**使用场景**：当训练数据只存储文档 ID 而非实际图片时（例如大规模数据集节省内存），通过 Corpus 在 `__getitem__` 时按需加载。

#### 5.1.2 ColPaliEngineDataset —— 核心数据集类

```python
# 文件: colpali_engine/data/dataset.py

Document = Union[str, Image.Image]  # 文档可以是文本或 PIL 图像

class ColPaliEngineDataset(Dataset):
    # 固定的输出键名
    QUERY_KEY = "query"
    POS_TARGET_KEY = "pos_target"
    NEG_TARGET_KEY = "neg_target"

    def __init__(
        self,
        data: List[Dict[str, Any]],       # 原始数据
        corpus: Optional[Corpus] = None,   # 可选的外部文档库
        query_column_name: str = "query",
        pos_target_column_name: str = "pos_target",
        neg_target_column_name: str = None,
        num_negatives: int = 3,            # 最大负样本数（防止OOM）
    ):
        self.data = data
        self.corpus = corpus
        ...
```

**`__getitem__()` —— 每次取一个训练样本**：

```python
def __getitem__(self, idx: int) -> Dict[str, Any]:
    sample = self.data[idx]

    query = sample[self.query_column_name]
    # 查询：字符串（如 "What is the hourly rate?"）

    pos_targets = sample[self.pos_target_column_name]
    if not isinstance(pos_targets, list):
        pos_targets = [pos_targets]
    # 正样本：一张或多张图片

    if self.neg_target_column_name is not None:
        neg_targets = sample[self.neg_target_column_name]
        if not isinstance(neg_targets, list):
            neg_targets = [neg_targets]
    else:
        neg_targets = None
    # 负样本：可选，来自数据中预计算的难负样本

    # 如果有外部 Corpus，通过 ID 检索实际文档
    if self.corpus is not None:
        pos_targets = [self.corpus.retrieve(doc_id) for doc_id in pos_targets]
        if neg_targets is not None:
            # 限制负样本数量，避免 CPU 内存溢出
            if len(neg_targets) > self.num_negatives:
                neg_targets = random.sample(neg_targets, self.num_negatives)
            neg_targets = [self.corpus.retrieve(doc_id) for doc_id in neg_targets]

    return {
        self.QUERY_KEY: query,           # str 或 List[str]
        self.POS_TARGET_KEY: pos_targets, # List[Image] 或 List[str]
        self.NEG_TARGET_KEY: neg_targets,  # List[Image] 或 None
    }
```

**数据流**：
```
HuggingFace Dataset / List[Dict]
    ↓ ColPaliEngineDataset.__getitem__(idx)
    ↓
{"query": "What is...", "pos_target": [PIL.Image], "neg_target": [PIL.Image, ...]}
    ↓ VisualRetrieverCollator.__call__(batch_of_dicts)
    ↓
{"query_input_ids": tensor, "query_attention_mask": tensor,
 "doc_input_ids": tensor, "doc_pixel_values": tensor, ...}
    ↓ Model.forward(**query_inputs), Model.forward(**doc_inputs)
    ↓
query_embeddings: (B, N_q, 128), doc_embeddings: (B, N_d, 128)
```

---

### 5.2 VisualRetrieverCollator

> **代码位置**：`colpali_engine/collators/visual_retriever_collator.py`

Collator 是 PyTorch DataLoader 的核心组件——它接收一批 Dataset 返回的字典，将它们整理成模型可以直接消费的批量张量。

```python
# 文件: colpali_engine/collators/visual_retriever_collator.py

N_AUGMENTATION_TOKENS = 10  # 查询增强 token 数量

class VisualRetrieverCollator:
    # 三种键前缀，用于区分查询、正文档、负文档
    query_prefix = "query_"
    pos_doc_prefix = "doc_"
    neg_doc_prefix = "neg_doc_"

    def __init__(
        self,
        processor: BaseVisualRetrieverProcessor,
        max_length: int = 2048,
    ):
        self.processor = processor
        self.max_length = max_length

        # ColPali 使用右填充（生成式模型的常规做法）
        if isinstance(self.processor, ColPaliProcessor):
            if self.processor.tokenizer.padding_side != "right":
                self.processor.tokenizer.padding_side = "right"
```

#### 5.2.1 `__call__()` —— 批量整理

```python
def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    queries = []
    pos_targets = []
    neg_targets = []

    # ===== 步骤1: 解析每个样本 =====
    for example in examples:
        query = example[ColPaliEngineDataset.QUERY_KEY]
        # 如果 query 是列表（多个可能的问法），随机选一个
        sampled_query = random.choice(query) if isinstance(query, list) else query
        queries.append(sampled_query)

        pos_tgt = example[ColPaliEngineDataset.POS_TARGET_KEY]
        sample_pos = random.choice(pos_tgt) if isinstance(pos_tgt, list) else pos_tgt
        pos_targets.append(sample_pos)

        neg_tgt = example.get(ColPaliEngineDataset.NEG_TARGET_KEY, None)
        if neg_tgt is not None:
            neg_targets.append(neg_tgt)

    # ===== 步骤2: 处理查询 —— 加前缀 + 增强 token =====
    queries = [
        self.processor.query_prefix
        + q
        + self.processor.query_augmentation_token * N_AUGMENTATION_TOKENS
        for q in queries
    ]
    # 例如 ColPali:
    #   query_prefix = ""
    #   query_augmentation_token = pad_token
    #   最终: "What is the hourly rate?<pad><pad><pad>...<pad>"（10个pad）
    #
    # 例如 ColQwen2:
    #   query_prefix = ""
    #   query_augmentation_token = "<|endoftext|>"
    #   最终: "What is the hourly rate?<|endoftext|>...<|endoftext|>"

    batch_query = self.auto_collate(queries, key_prefix=self.query_prefix)
    # 返回: {"query_input_ids": tensor, "query_attention_mask": tensor}

    # ===== 步骤3: 处理正文档 =====
    batch_pos_target = self.auto_collate(pos_targets, key_prefix=self.pos_doc_prefix)
    # 如果是图片 → processor.process_images()
    # 返回: {"doc_input_ids": ..., "doc_pixel_values": ..., "doc_attention_mask": ...}

    # ===== 步骤4: 处理负文档（如果有） =====
    batch_neg_target = (
        self.auto_collate(neg_targets, key_prefix=self.neg_doc_prefix)
        if neg_targets else {}
    )

    return {**batch_query, **batch_pos_target, **batch_neg_target}
```

#### 5.2.2 `auto_collate()` —— 自动识别文本/图像并处理

```python
def auto_collate(self, batch: List[Union[str, Image]], key_prefix: str = "") -> Dict[str, Any]:
    if isinstance(batch[0], str):
        proc_batch = self.processor.process_texts(texts=batch)
    elif isinstance(batch[0], Image):
        proc_batch = self.processor.process_images(images=batch)
    elif isinstance(batch[0], list):
        # 处理负样本列表（每个样本有多个负例）
        if isinstance(batch[0][0], str):
            batch_size = len(batch)
            all_texts = [text for texts in batch for text in texts]
            num_negatives = len(all_texts) // batch_size
            proc_batch = self.processor.process_texts(texts=all_texts)
        elif isinstance(batch[0][0], Image):
            batch_size = len(batch)
            all_imgs = [img for imgs in batch for img in imgs]
            num_negatives = len(all_imgs) // batch_size
            proc_batch = self.processor.process_images(images=all_imgs)
        # 将扁平化的张量重新reshape为 (batch_size, num_negatives, ...)
        for k, v in proc_batch.items():
            if isinstance(v, torch.Tensor):
                proc_batch[k] = v.view(batch_size, num_negatives, *v.shape[1:])

    return prefix_keys(proc_batch, key_prefix)
    # prefix_keys: 给所有键加前缀，如 "input_ids" → "query_input_ids"
```

**为什么要用 `prefix_keys`？** 因为查询和文档共用同一个模型，它们的输入键名相同（都是 `input_ids`、`attention_mask` 等）。加前缀后可以放在同一个字典中而不冲突，训练器根据前缀分离查询和文档的输入。

---

### 5.3 SingleDatasetBatchSampler

> **代码位置**：`colpali_engine/data/sampler.py`

当训练使用**多个数据集**时（如 DocVQA + InfoVQA + ArxivQA），这个采样器确保**每个 batch 只来自一个数据集**。

```python
# 文件: colpali_engine/data/sampler.py

class SingleDatasetBatchSampler(BatchSampler):
    """
    每个 batch 只从一个数据集采样，数据集的选择概率与其剩余样本数成正比。
    """

    def __init__(
        self,
        datasets: List[Dataset],
        global_batch_size: int,
        drop_last: bool = True,
        generator: Optional[torch.Generator] = None,
    ):
        self.datasets = datasets
        self.global_batch_size = global_batch_size
        self.dataset_sizes = [len(dataset) for dataset in datasets]
        # 累积大小，用于将局部索引转换为全局索引
        self.cumsum_sizes = np.cumsum([0] + self.dataset_sizes).tolist()
        # 例如: [0, 5000, 8000, 10000]
        # 数据集0: 索引 0~4999, 数据集1: 索引 5000~7999, ...

        # 为每个数据集创建打乱的索引
        self.indices_per_dataset = [
            torch.randperm(size, generator=self.generator).tolist()
            for size in self.dataset_sizes
        ]
```

**`__iter__()` —— 按概率选择数据集并产出 batch 索引**：

```python
def __iter__(self) -> Iterator[List[int]]:
    self.current_positions = [0] * len(self.datasets)
    self.available_datasets = list(range(len(self.datasets)))

    while self.available_datasets:
        # 根据剩余样本数计算选择概率
        lengths = [self.current_data_lengths[i] for i in self.available_datasets]
        total_length = sum(lengths)
        probs = torch.tensor(lengths, dtype=torch.float) / total_length

        # 按概率抽选一个数据集
        dataset_idx_in_available = torch.multinomial(probs, num_samples=1, ...).item()
        dataset_idx = self.available_datasets[dataset_idx_in_available]

        # 从该数据集取 global_batch_size 个样本
        current_pos = self.current_positions[dataset_idx]
        end_pos = current_pos + self.global_batch_size

        if end_pos <= self.max_positions[dataset_idx]:
            batch_indices = [
                idx + self.cumsum_sizes[dataset_idx]  # 局部索引 → 全局索引
                for idx in dataset_indices[current_pos:end_pos]
            ]
            yield batch_indices
```

**为什么每个 batch 只用一个数据集？** 不同数据集的图像分辨率、文本长度可能差异很大。混合采样会导致 padding 严重、计算效率下降。单数据集采样保证了 batch 内的样本相对均匀。

---

## 6. 训练流程

### 6.1 ColModelTraining —— 训练入口

> **代码位置**：`colpali_engine/trainer/colmodel_training.py`

这是训练的最高层封装，整合了模型、数据、损失函数和训练参数。

#### 6.1.1 配置类

```python
# 文件: colpali_engine/trainer/colmodel_training.py

@dataclass
class ColModelTrainingConfig:
    model: Union[PreTrainedModel, PeftModel]     # ColPali / ColQwen2 等
    processor: BaseVisualRetrieverProcessor       # 对应的 Processor
    train_dataset: Union[ColPaliEngineDataset, List[ColPaliEngineDataset]]
    eval_dataset: Optional[Union[ColPaliEngineDataset, Dict[str, ColPaliEngineDataset]]] = None
    tr_args: Optional[TrainingArguments] = None   # HuggingFace TrainingArguments
    max_length: int = 256                         # 最大序列长度
    peft_config: Optional[LoraConfig] = None      # LoRA 配置
    loss_func: Optional[Callable] = ColbertLoss() # 默认使用 ColbertLoss
    pretrained_peft_model_name_or_path: Optional[str] = None
```

#### 6.1.2 PEFT/LoRA 自动配置

```python
def __post_init__(self):
    # 自动设置输出目录
    if self.output_dir is None:
        sanitized_name = str(self.model.name_or_path).replace("/", "_")
        self.output_dir = f"./models/{sanitized_name}"

    # ===== LoRA 配置 =====
    if self.peft_config is not None:
        if self.pretrained_peft_model_name_or_path is None:
            # 新建 LoRA 适配器
            self.model = get_peft_model(self.model, self.peft_config)
            self.model.print_trainable_parameters()
            # 输出示例: "trainable params: 6,553,600 || all params: 3,000,000,000 || trainable%: 0.22"
        else:
            print(f"Adapter already loaded from {self.pretrained_peft_model_name_or_path}")
```

**LoRA 的作用**：论文中提到使用 LoRA (r=32, α=32) 只训练语言模型的 Transformer 层和投影层。这大幅减少了可训练参数（从 ~30 亿降到 ~600 万），使得在消费级 GPU 上也能训练。

#### 6.1.3 训练启动

```python
# 文件: colpali_engine/trainer/colmodel_training.py

class ColModelTraining:
    def __init__(self, config: ColModelTrainingConfig):
        self.config = config
        self.model = config.model
        # 创建 Collator
        self.collator = VisualRetrieverCollator(
            processor=config.processor,
            max_length=config.max_length,
        )

    def train(self) -> None:
        # 创建训练器并启动
        trainer = ContrastiveTrainer(
            model=self.model,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            args=self.config.tr_args,
            data_collator=self.collator,
            loss_func=self.config.loss_func,
            is_vision_model=True,
        )
        trainer.args.remove_unused_columns = False
        result = trainer.train(
            resume_from_checkpoint=self.config.tr_args.resume_from_checkpoint
        )
        print_summary(result)

    def save(self):
        self.model.save_pretrained(self.config.output_dir)
        self.config.processor.save_pretrained(self.config.output_dir)
        # 保存 git commit hash 用于实验追踪
        with open(f"{self.config.output_dir}/git_hash.txt", "w") as f:
            f.write(self.current_git_hash)
```

**训练脚本调用链**：

```
scripts/train/train_colbert.py
    ↓ 加载 YAML 配置
    ↓ configue.load(config_file)
    ↓
ColModelTrainingConfig(model=ColQwen2, processor=ColQwen2Processor, ...)
    ↓
ColModelTraining(config)
    ↓
ColModelTraining.train()
    ↓
ContrastiveTrainer(...)
    ↓
trainer.train()  ← HuggingFace Trainer 的标准训练循环
```

---

### 6.2 ContrastiveTrainer —— 对比学习训练器

> **代码位置**：`colpali_engine/trainer/contrastive_trainer.py`

继承自 HuggingFace 的 `Trainer`，重写了损失计算和数据加载逻辑。

#### 6.2.1 初始化

```python
# 文件: colpali_engine/trainer/contrastive_trainer.py

class ContrastiveTrainer(Trainer):
    def __init__(self, loss_func, is_vision_model, compute_symetric_loss=False, *args, **kwargs):
        # 如果传入多个数据集列表，先拼接成 ConcatDataset
        if isinstance(kwargs["train_dataset"], list):
            train_dataset_list = kwargs["train_dataset"]
            kwargs["train_dataset"] = concat_datasets(
                train_dataset_list, batch_size=kwargs["args"].train_batch_size
            )
        else:
            train_dataset_list = None

        super().__init__(*args, **kwargs)
        self.loss_func = loss_func
        self.args.remove_unused_columns = False  # 保留所有列
        self.train_dataset_list = train_dataset_list
```

#### 6.2.2 compute_loss() —— 核心损失计算

这是整个训练循环中**最关键的函数**，完整的 forward + loss 计算都在这里。

```python
# 文件: colpali_engine/trainer/contrastive_trainer.py

def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    # ===== 步骤1: 分离查询输入 =====
    query_inputs = {
        k[len(self.query_prefix):]: v
        for k, v in inputs.items()
        if k.startswith(self.query_prefix)
    }
    # 例如: {"query_input_ids": ..., "query_attention_mask": ...}
    # → {"input_ids": ..., "attention_mask": ...}

    # ===== 步骤2: 查询前向传播 =====
    query_outputs = model(**query_inputs)
    # query_outputs: (batch_size, query_length, 128) — 多向量嵌入

    # ===== 步骤3: 分离文档输入 =====
    doc_inputs = {
        k[len(self.pos_prefix):]: v
        for k, v in inputs.items()
        if k.startswith(self.pos_prefix)
    }
    # {"doc_input_ids": ..., "doc_pixel_values": ..., "doc_attention_mask": ...}
    # → {"input_ids": ..., "pixel_values": ..., "attention_mask": ...}

    # ===== 步骤4: 文档前向传播 =====
    doc_outputs = model(**doc_inputs)
    # doc_outputs: (batch_size, doc_length, 128) — 多向量嵌入

    # ===== 步骤5: 处理显式负样本（如果有） =====
    if "neg_doc_input_ids" in inputs:
        num_negs = inputs["neg_doc_input_ids"].size(1)
        neg_doc_inputs = self._reshape_neg_doc_inputs(inputs)
        # 从 (batch, num_negs, seq_len) 展平为 (batch*num_negs, seq_len)
        neg_doc_outputs = model(**neg_doc_inputs)
        neg_doc_outputs = self._reshape_neg_doc_outputs(neg_doc_outputs, num_negs)
        # 再 reshape 回 (batch, num_negs, seq_len, 128)
    else:
        neg_doc_outputs = None

    # ===== 步骤6: 计算损失 =====
    loss = self._compute_loss_from_outputs(query_outputs, doc_outputs, neg_doc_outputs)

    # ===== 步骤7 (可选): 对称损失 =====
    if self.compute_symetric_loss:
        sym_loss = self._compute_loss_from_outputs(doc_outputs, query_outputs)
        loss = (loss + sym_loss) / 2

    return (loss, (query_outputs, doc_outputs)) if return_outputs else loss
```

#### 6.2.3 多 GPU 分布式处理

```python
def _compute_loss_from_outputs(self, query_outputs, pos_target_outputs, neg_target_outputs=None):
    offset = 0
    batch_size = query_outputs.size(0)

    if self.accelerator.num_processes > 1 and self.accelerator.sync_gradients:
        # ===== 多GPU: gather 所有进程的文档嵌入 =====
        pos_target_outputs = self.accelerator.pad_across_processes(
            pos_target_outputs, dim=1, pad_index=0, pad_first=True
        )
        # pad_across_processes: 确保所有 GPU 上的张量形状一致
        # dim=1 是 seq_len 维度（不同文档可能有不同的 patch 数）
        # pad_first=True: 在前面 pad（左填充）

        pos_target_outputs = concat_all_gather(pos_target_outputs)
        # all_gather: 收集所有 GPU 的文档嵌入
        # 从 (local_B, seq_len, 128) → (global_B, seq_len, 128)
        # 保持了梯度！（使用 torch.distributed.nn.functional.all_gather）

        rank = self.accelerator.process_index
        offset = rank * batch_size
        # GPU-0: offset=0, GPU-1: offset=32, GPU-2: offset=64, ...

    # ===== 调用损失函数 =====
    if neg_target_outputs is not None:
        loss = self.loss_func(
            query_embeddings=query_outputs,
            doc_embeddings=pos_target_outputs,  # 可能已 gather
            neg_doc_embeddings=neg_target_outputs,
            offset=offset,
        )
    else:
        loss = self.loss_func(
            query_embeddings=query_outputs,
            doc_embeddings=pos_target_outputs,
            offset=offset,
        )
    return loss
```

**`concat_all_gather` 的实现**：

```python
# 文件: colpali_engine/trainer/contrastive_trainer.py

def concat_all_gather(t: torch.Tensor) -> torch.Tensor:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.cat(all_gather(t), dim=0)  # 保持梯度图
    return t
```

**为什么要 all_gather 文档嵌入但不 gather 查询嵌入？**

这是对比学习的经典技巧：增大有效负样本池。每个 GPU 上的查询只 gather 所有 GPU 的文档嵌入作为候选，但查询本身不需要 gather（每个 GPU 独立计算自己的损失）。这样有效负样本数从 local_batch_size 增大到 global_batch_size（= local_batch_size × num_gpus），显著改善训练效果。

---

### 6.3 ColModelTorchTraining —— 原生 PyTorch 分布式训练

> **代码位置**：`colpali_engine/trainer/colmodel_torch_training.py`

这是一个**不依赖 HuggingFace Trainer**的原生 PyTorch 训练循环实现，提供了更细粒度的控制。

#### 6.3.1 初始化

```python
# 文件: colpali_engine/trainer/colmodel_torch_training.py

class ColModelTorchTraining:
    def __init__(self, config: ColModelTrainingConfig):
        ...
        # 初始化分布式
        if dist.is_available() and not dist.is_initialized():
            dist.init_process_group(backend="nccl", init_method="env://")
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))

        # 将模型放到对应 GPU
        device = torch.device(f"cuda:{self.local_rank}")
        torch.cuda.set_device(device)
        self.model.to(device)

        # 可选：梯度检查点（以时间换空间）
        if getattr(self.config.tr_args, "gradient_checkpointing", False):
            self.model.gradient_checkpointing_enable(...)

        # DDP 包装
        self.model = DistributedDataParallel(
            self.model, device_ids=[self.local_rank], output_device=self.local_rank
        )

        # torch.compile 优化（PyTorch 2.x 特性）
        self.model = torch.compile(self.model, backend="inductor", dynamic=True)
```

#### 6.3.2 训练循环核心

```python
def train(self) -> None:
    # ===== 混合精度设置 =====
    use_amp = getattr(self.config, "use_amp", False)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # ===== DataLoader =====
    sampler = DistributedSampler(self.train_dataset) if dist.is_initialized() else None
    train_loader = DataLoader(
        self.train_dataset,
        batch_size=self.config.tr_args.per_device_train_batch_size,
        sampler=sampler,
        collate_fn=self.collator,
        num_workers=...,
        pin_memory=True,
        drop_last=True,
    )

    # ===== 优化器 + 学习率调度 =====
    optimizer = torch.optim.AdamW(
        self.model.parameters(),
        lr=self.config.tr_args.learning_rate,
        weight_decay=self.config.tr_args.weight_decay,
    )

    # 线性预热 + 线性衰减
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, num_training_steps - warmup_steps))
        return max(0.1, 1.0 - (1.0 - 0.1) * progress)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

#### 6.3.3 关键：跨 GPU Padding 同步

```python
# 文件: colpali_engine/trainer/colmodel_torch_training.py (在 train() 内部)

def pad_to_max_len_right(x: torch.Tensor) -> torch.Tensor:
    """
    右 pad x 的 dim=1（seq_len），使所有 rank 上的张量具有相同长度。
    这是 all_gather 的前置要求。

    问题：不同 GPU 上的图片可能有不同数量的 patch，
    导致 doc_embeddings 的 seq_len 不一致，无法直接 gather。

    解决：先 all_reduce 获取全局最大 seq_len，然后 pad 到一致。
    """
    local_len = x.size(1)  # 本地 seq_len
    len_tensor = torch.tensor(local_len, device=x.device)
    dist.all_reduce(len_tensor, op=dist.ReduceOp.MAX)  # 所有 GPU 取 max
    max_len = len_tensor.item()

    if local_len < max_len:
        pad_amount = max_len - local_len
        x = torch.nn.functional.pad(x, (0, 0, 0, pad_amount), value=0.0)
        # pad 格式: (D_left, D_right, L_left, L_right)
        # 在 dim=1 的 right 端 pad 0
    return x

# 使用：
d_embed = pad_to_max_len_right(d_embed)   # 统一文档嵌入的序列长度
d_global = gather_with_grad(d_embed)       # 安全 gather

# 计算损失 —— 使用 offset 标识正样本位置
loss = loss_fn(
    q_embed, d_global,
    offset=(dist.get_rank() * batch["query_input_ids"].shape[0])
)
```

#### 6.3.4 `gather_with_grad` —— 保持梯度的 all_gather

```python
def gather_with_grad(x: torch.Tensor) -> torch.Tensor:
    return all_gather_tensor_autograd(x, gather_dim=0, group=dist.group.WORLD)
```

`all_gather_tensor_autograd` 是 PyTorch 2.x 的功能，与普通 `all_gather` 不同的是——**它在反向传播时会正确传播梯度**到所有参与 gather 的进程。这对对比学习至关重要，因为负样本的文档嵌入也需要接收梯度更新。

### 6.4 完整训练数据流总结

```
                    训练数据流
                    =========

┌─────────────┐   ┌──────────────────┐   ┌────────────────────┐
│ HF Dataset  │ → │ ColPaliEngine    │ → │ VisualRetriever    │
│ (query,     │   │ Dataset          │   │ Collator           │
│  image_id,  │   │ .__getitem__()   │   │ .__call__()        │
│  neg_ids)   │   │ → {query, img,   │   │ → tokenize/process │
│             │   │    neg_imgs}     │   │ → add prefix keys   │
└─────────────┘   └──────────────────┘   └────────────────────┘
                                                  │
                   ┌──────────────────────────────┤
                   ↓                              ↓
    ┌───────────────────────┐    ┌─────────────────────────────┐
    │ query_input_ids       │    │ doc_input_ids               │
    │ query_attention_mask  │    │ doc_pixel_values            │
    └──────────┬────────────┘    │ doc_attention_mask          │
               │                 │ doc_image_grid_thw (Qwen2)  │
               ↓                 └──────────────┬──────────────┘
    ┌────────────────────┐              ┌───────┤
    │ model(**query)     │              │ model(**doc)
    │ → (B, N_q, 128)   │              │ → (B, N_d, 128)
    └──────────┬─────────┘              └───────┬──────────────┐
               │                                │   all_gather │
               │                                ↓              │
               │                    ┌────────────────────┐     │
               │                    │ (B*GPUs, N_d, 128) │     │
               │                    └────────┬───────────┘     │
               │                             │                 │
               └─────────────┬───────────────┘
                             ↓
                  ┌──────────────────────┐
                  │ loss_func(           │
                  │   query_embeddings,  │
                  │   doc_embeddings,    │
                  │   offset             │
                  │ )                    │
                  └──────────┬───────────┘
                             ↓
                       scalar loss
                             ↓
                       loss.backward()
                             ↓
                       optimizer.step()
```

---

## 7. 推理与评分

### 7.1 MaxSim 评分函数

> **代码位置**：`colpali_engine/utils/processing_utils.py`

推理时的评分函数实现了论文中的延迟交互公式 $LI(q,d)$，与训练时损失函数中的 einsum 操作本质相同，但做了工程优化以支持大规模语料库。

#### 7.1.1 `score_multi_vector()` —— 核心 MaxSim 评分

```python
# 文件: colpali_engine/utils/processing_utils.py

@staticmethod
def score_multi_vector(
    qs: Union[torch.Tensor, List[torch.Tensor]],  # 查询嵌入列表
    ps: Union[torch.Tensor, List[torch.Tensor]],  # 文档嵌入列表
    batch_size: int = 128,                          # 分批计算避免OOM
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    """
    计算 ColBERT 风格的 MaxSim 分数。
    输入是变长的多向量嵌入列表，输出是 (n_queries, n_passages) 的分数矩阵。
    """
    device = device or get_torch_device("auto")
    scores_list: List[torch.Tensor] = []

    # ===== 外循环: 遍历查询批次 =====
    for i in range(0, len(qs), batch_size):
        scores_batch = []

        # pad 查询到相同长度（不同查询可能有不同的 token 数）
        qs_batch = torch.nn.utils.rnn.pad_sequence(
            qs[i : i + batch_size], batch_first=True, padding_value=0
        ).to(device)
        # qs_batch: (min(batch_size, remaining), max_query_len, 128)

        # ===== 内循环: 遍历文档批次 =====
        for j in range(0, len(ps), batch_size):
            ps_batch = torch.nn.utils.rnn.pad_sequence(
                ps[j : j + batch_size], batch_first=True, padding_value=0
            ).to(device)
            # ps_batch: (min(batch_size, remaining), max_doc_len, 128)

            scores_batch.append(
                torch.einsum("bnd,csd->bcns", qs_batch, ps_batch)
                #   b: 查询batch, n: 查询token, d: dim
                #   c: 文档batch, s: 文档token
                # → (B_q, B_d, N_q, N_d): 所有 token 对的点积
                .max(dim=3)[0]
                # 对每个查询token，取最匹配的文档token
                # → (B_q, B_d, N_q)
                .sum(dim=2)
                # 求和所有查询token的最大匹配分数
                # → (B_q, B_d): 这就是 LI(q,d)
            )

        scores_batch = torch.cat(scores_batch, dim=1).cpu()
        # 拼接所有文档批次的分数 → (B_q, total_docs)
        scores_list.append(scores_batch)

    scores = torch.cat(scores_list, dim=0)
    # 拼接所有查询批次 → (total_queries, total_docs)

    return scores.to(torch.float32)
```

**工程细节**：

1. **分批计算**：如果一次性计算所有查询和文档的交互矩阵，内存会爆炸。分批处理每次只计算 `batch_size × batch_size` 的块。
2. **`pad_sequence`**：不同查询/文档有不同的 token 数，先 pad 到批次内最长，padding 部分的嵌入是 0（L2 归一化后被 mask 过），不影响 MaxSim 结果。
3. **结果移回 CPU**：大规模检索时，分数矩阵可能很大，放在 GPU 上会占用宝贵的显存。

#### 7.1.2 `score_single_vector()` —— 双编码器评分

```python
@staticmethod
def score_single_vector(
    qs: Union[torch.Tensor, List[torch.Tensor]],
    ps: Union[torch.Tensor, List[torch.Tensor]],
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    """
    双编码器的点积评分，比 MaxSim 简单得多。
    """
    device = device or get_torch_device("auto")
    if isinstance(qs, list):
        qs = torch.stack(qs).to(device)
        ps = torch.stack(ps).to(device)

    scores = torch.einsum("bd,cd->bc", qs, ps)
    # qs: (N_q, D), ps: (N_d, D) → scores: (N_q, N_d)
    # 每个查询和每个文档直接做点积（因为已经L2归一化，等价于余弦相似度）

    return scores.to(torch.float32)
```

#### 7.1.3 `process_queries()` —— 查询预处理（推理时）

```python
# 文件: colpali_engine/utils/processing_utils.py

def process_queries(
    self,
    texts: Optional[List[str]] = None,
    queries: Optional[List[str]] = None,
    suffix: Optional[str] = None,
) -> Union[BatchFeature, BatchEncoding]:
    """推理时处理查询文本，自动添加前缀和增强 token。"""
    if suffix is None:
        suffix = self.query_augmentation_token * 10
        # 默认添加 10 个增强 token

    texts = [self.query_prefix + text + suffix for text in texts]
    return self.process_texts(texts=texts)
```

---

### 7.2 FastPlaid 索引加速

> **代码位置**：`colpali_engine/utils/processing_utils.py`

对于大规模语料库（百万级别），暴力 MaxSim 计算太慢。FastPlaid 是一个近似最近邻搜索库，专为 ColBERT 风格的多向量检索设计。

```python
# 文件: colpali_engine/utils/processing_utils.py

@staticmethod
def create_plaid_index(
    ps: Union[torch.Tensor, List[torch.Tensor]],
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    """
    从文档嵌入构建 FastPlaid 索引。
    这个索引支持快速近似搜索，避免暴力遍历所有文档。
    """
    fast_plaid_index = search.FastPlaid(index="index")
    device = device or get_torch_device("auto")
    fast_plaid_index.create(
        documents_embeddings=[d.to(device).to(torch.float32) for d in ps]
    )
    return fast_plaid_index

@staticmethod
def get_topk_plaid(
    qs: Union[torch.Tensor, List[torch.Tensor]],
    plaid_index,
    k: int = 10,
    batch_size: int = 128,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    """
    使用 FastPlaid 索引快速检索 top-k 文档。
    """
    scores_list = []
    for i in range(0, len(qs), batch_size):
        qs_batch = torch.nn.utils.rnn.pad_sequence(
            qs[i : i + batch_size], batch_first=True, padding_value=0
        ).to(device)
        scores_batch = plaid_index.search(
            queries_embeddings=qs_batch.to(torch.float32), top_k=k
        )
        scores_list.append(scores_batch)
    return scores_list
```

**推理流程总结**：

```
离线阶段（索引构建）:
  文档页面图像 → Processor.process_images() → Model.forward()
  → doc_embeddings: List[(N_d_i, 128)]
  → Optional: create_plaid_index(doc_embeddings) → Fast索引

在线阶段（查询匹配）:
  查询文本 → Processor.process_queries() → Model.forward()
  → query_embeddings: List[(N_q, 128)]

  方案A（小规模，精确）:
    score_multi_vector(query_embeddings, doc_embeddings) → (N_q, N_docs) 分数矩阵
    → .topk(k) → 返回最相关文档

  方案B（大规模，近似）:
    get_topk_plaid(query_embeddings, plaid_index, k) → top-k 文档
```

---

## 8. 可解释性模块

> **代码位置**：`colpali_engine/interpretability/`

ColPali 的一个独特优势是可解释性——可以通过延迟交互热图直观地看到**查询中的每个词匹配了文档页面的哪个区域**。论文 5.3 节专门展示了这一特性。

### 8.1 生成相似度图

> **代码位置**：`colpali_engine/interpretability/similarity_map_utils.py`

```python
# 文件: colpali_engine/interpretability/similarity_map_utils.py

def get_similarity_maps_from_embeddings(
    image_embeddings: torch.Tensor,    # (batch_size, image_tokens, dim)
    query_embeddings: torch.Tensor,    # (batch_size, query_tokens, dim)
    n_patches: Union[Tuple[int, int], List[Tuple[int, int]]],
    image_mask: torch.Tensor,          # (batch_size, image_tokens)
) -> List[torch.Tensor]:
    """
    计算查询嵌入和图像嵌入之间的相似度图。
    返回列表中的每个元素形状为 (query_tokens, n_patches_x, n_patches_y)。
    """

    if isinstance(n_patches, tuple):
        n_patches = [n_patches] * image_embeddings.size(0)
        # 如果所有图像具有相同的 patch 网格，广播

    similarity_maps: List[torch.Tensor] = []

    for idx in range(image_embeddings.size(0)):
        # ===== 验证 =====
        if image_mask[idx].sum() != n_patches[idx][0] * n_patches[idx][1]:
            raise ValueError(
                f"patch 数量 ({n_patches[idx][0]} x {n_patches[idx][1]}) "
                f"与非 padding 图像 token 数 ({image_mask[idx].sum()}) 不匹配"
            )

        # ===== 将 1D patch 序列重排为 2D 空间网格 =====
        image_embedding_grid = rearrange(
            image_embeddings[idx][image_mask[idx]],
            # 只取图像 token（排除文本 prompt token）
            # 形状: (n_patches_x * n_patches_y, dim)
            "(h w) c -> w h c",
            w=n_patches[idx][0],
            h=n_patches[idx][1],
        )
        # 形状: (n_patches_x, n_patches_y, dim)
        #
        # rearrange 来自 einops 库:
        #   "(h w) c -> w h c" 将扁平化的 patch 序列还原为 2D 网格
        #   h: patch 行数, w: patch 列数, c: 嵌入维度

        # ===== 计算每个查询 token 与每个 patch 的相似度 =====
        similarity_map = torch.einsum(
            "nk,ijk->nij", query_embeddings[idx], image_embedding_grid
        )
        # einsum 解析:
        #   n: query token 索引
        #   k: 嵌入维度（共享）
        #   i: patch x 坐标
        #   j: patch y 坐标
        #
        # 输入:
        #   query_embeddings[idx]: (N_q, D)     — 一个查询的所有 token
        #   image_embedding_grid:  (W, H, D)    — 2D patch 网格
        #
        # 输出:
        #   similarity_map: (N_q, W, H)
        #   含义: 第 n 个查询 token 与位于 (i,j) 位置的 patch 的点积
        #   值越大 → 该 patch 与该查询 token 越相关

        similarity_maps.append(similarity_map)

    return similarity_maps
```

**张量变化**：
```
image_embeddings: (batch, image_tokens, 128)
    ↓ image_mask 过滤
    ↓ rearrange "(h w) c -> w h c"
image_embedding_grid: (n_patches_x, n_patches_y, 128)
    ↓ einsum "nk,ijk->nij" with query_embeddings
similarity_map: (query_tokens, n_patches_x, n_patches_y)
```

### 8.2 归一化相似度图

```python
# 文件: colpali_engine/interpretability/similarity_map_utils.py

def normalize_similarity_map(
    similarity_map: torch.Tensor,
    value_range: Optional[Tuple[float, float]] = None,
) -> torch.Tensor:
    """
    将相似度图归一化到 [0, 1] 范围，用于可视化。
    """
    if value_range is None:
        # 自动计算 min/max
        min_vals = similarity_map.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
        max_vals = similarity_map.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
    else:
        min_vals, max_vals = value_range
        # 使用指定范围（跨多个查询 token 共享同一个范围时有用）

    # 归一化: (x - min) / (max - min + ε)
    similarity_map_normalized = (similarity_map - min_vals) / (max_vals - min_vals + EPSILON)
    # EPSILON = 1e-10，防止除零

    return similarity_map_normalized
```

### 8.3 可视化热图

> **代码位置**：`colpali_engine/interpretability/similarity_maps.py`

```python
# 文件: colpali_engine/interpretability/similarity_maps.py

def plot_similarity_map(
    image: Image.Image,
    similarity_map: torch.Tensor,  # (n_patches_x, n_patches_y)
    figsize: Tuple[int, int] = (8, 8),
    show_colorbar: bool = False,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    将相似度图叠加在原始图像上，生成热图可视化。
    """
    # 转为 numpy 数组
    img_array = np.array(image.convert("RGBA"))

    # 归一化并转为 PIL 图像
    similarity_map_array = normalize_similarity_map(similarity_map).to(torch.float32).cpu().numpy()
    # (n_patches_x, n_patches_y) → 值在 [0, 1]

    # 调整维度和分辨率
    similarity_map_array = rearrange(similarity_map_array, "h w -> w h")
    similarity_map_image = Image.fromarray(
        (similarity_map_array * 255).astype("uint8")
    ).resize(image.size, Image.Resampling.BICUBIC)
    # 将低分辨率的 patch 相似度图上采样到原始图像分辨率

    # 创建叠加可视化
    with plt.style.context("dark_background"):
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(img_array)
        im = ax.imshow(
            similarity_map_image,
            cmap=sns.color_palette("mako", as_cmap=True),  # 使用 mako 颜色映射
            alpha=0.5,  # 半透明叠加
        )
        ax.set_axis_off()
        fig.tight_layout()

    return fig, ax
```

**`plot_all_similarity_maps()` 为查询中的每个 token 生成一张热图**：

```python
def plot_all_similarity_maps(
    image: Image.Image,
    query_tokens: List[str],
    similarity_maps: torch.Tensor,  # (query_tokens, n_patches_x, n_patches_y)
    normalize_per_query: bool = True,
) -> List[Tuple[plt.Figure, plt.Axes]]:
    """
    可视化每个查询 token 的相似度图。
    例如查询 "What is the hourly rate?"
    → 生成 "What" 的热图、"is" 的热图、"hourly" 的热图、...
    """
    ...
```

### 8.4 可解释性使用示例

```python
# 伪代码 - 典型使用流程
from colpali_engine.interpretability import get_similarity_maps_from_embeddings, plot_all_similarity_maps

# 1. 获取嵌入
with torch.no_grad():
    image_embeddings = model(**batch_images)      # (1, N_d, 128)
    query_embeddings = model(**batch_queries)      # (1, N_q, 128)

# 2. 获取 patch 网格大小
n_patches = processor.get_n_patches(image_size, patch_size)
# 例如 (32, 32)

# 3. 获取图像 token mask
image_mask = processor.get_image_mask(batch_images)
# (1, seq_len) — True 表示图像 token

# 4. 计算相似度图
similarity_maps = get_similarity_maps_from_embeddings(
    image_embeddings, query_embeddings, n_patches, image_mask
)
# List of (N_q, 32, 32)

# 5. 可视化
query_tokens = processor.tokenizer.tokenize(query_text)
figs = plot_all_similarity_maps(image, query_tokens, similarity_maps[0])
```

---

## 9. Token 压缩模块

> **代码位置**：`colpali_engine/compression/token_pooling/`

论文 5.2 节讨论了 ColPali 的存储开销：每页约 257.5 KB（1030 个 128 维 float16 向量）。Token 压缩旨在减少存储的向量数量。

### 9.1 基类 BaseTokenPooler

```python
# 文件: colpali_engine/compression/token_pooling/base_token_pooling.py

@dataclass
class TokenPoolingOutput:
    pooled_embeddings: Union[List[torch.Tensor], torch.Tensor]
    # 池化后的嵌入
    cluster_id_to_indices: Optional[Dict[int, Tuple[torch.Tensor]]] = None
    # 可选：聚类 ID 到原始 token 索引的映射（用于可解释性）

class BaseTokenPooler(ABC):
    @abstractmethod
    def _pool_embeddings_impl(self, embeddings, num_workers, *args, **kwargs):
        pass

    def pool_embeddings(self, embeddings, return_dict=True, padding=False, ...):
        """
        公共接口：接受 List[2D] 或 3D tensor，返回 TokenPoolingOutput。
        """
        self._validate_embeddings(embeddings)
        prepared = self._prepare_embeddings(embeddings, padding=padding)
        pooled, cluster_ids = self._pool_embeddings_impl(prepared, ...)
        if return_dict:
            return TokenPoolingOutput(pooled, cluster_ids)
        return pooled
```

### 9.2 HierarchicalTokenPooler —— 层次聚类池化

```python
# 文件: colpali_engine/compression/token_pooling/hierarchical_token_pooling.py

class HierarchicalTokenPooler(BaseTokenPooler):
    """
    基于 token 嵌入相似度的层次聚类池化。
    相似的 patch 被合并为一个簇，簇内取均值作为代表。
    """

    def _pool_single_embedding(
        self,
        embedding: torch.Tensor,  # (token_length, embedding_dim)
        pool_factor: int,
    ) -> Tuple[torch.Tensor, Dict]:

        token_length = embedding.size(0)
        if pool_factor == 1:
            # pool_factor=1 意味着不做池化
            return embedding, {0: (torch.arange(token_length),)}

        # 转到 CPU（scipy 只支持 CPU）
        embedding = embedding.to(torch.float32).cpu()

        # ===== 步骤1: 计算 token 间相似度 =====
        similarities = torch.mm(embedding, embedding.t())
        # (token_length, token_length) — 余弦相似度（已L2归一化）
        distances = 1 - similarities.numpy()
        # 转为距离矩阵

        # ===== 步骤2: 层次聚类 =====
        Z = linkage(distances, metric="euclidean", method="ward")
        # scipy.cluster.hierarchy.linkage:
        #   "ward" 方法: 最小化簇内方差
        #   Z 是 linkage 矩阵，记录了合并顺序

        max_clusters = max(token_length // pool_factor, 1)
        # pool_factor=3: 1030 个 token → 最多 343 个簇
        # 减少了约 67% 的向量数量

        cluster_labels = fcluster(Z, t=max_clusters, criterion="maxclust") - 1
        # fcluster: 将 linkage 结果切割为 max_clusters 个簇
        # 返回每个 token 的簇标签 [0, 1, ..., max_clusters-1]

        # ===== 步骤3: 对每个簇取均值 =====
        for cluster_id in range(max_clusters):
            cluster_indices = torch.where(torch.tensor(cluster_labels == cluster_id))
            if cluster_indices[0].numel() > 0:
                pooled_embedding = embedding[cluster_indices].mean(dim=0)
                # 簇内所有 token 嵌入的平均值
                pooled_embedding = F.normalize(pooled_embedding, p=2, dim=-1)
                # 重新 L2 归一化
                list_pooled_embeddings.append(pooled_embedding)

        pooled_embeddings = torch.stack(list_pooled_embeddings, dim=0)
        # (num_clusters, embedding_dim)

        return pooled_embeddings, cluster_id_to_indices
```

**为什么用层次聚类而不是 K-means？**

1. 层次聚类不需要随机初始化，结果确定
2. 可以自然地产生不同粒度的聚类（通过改变 pool_factor）
3. Ward 方法在保持簇内紧凑性方面表现良好
4. 论文图 3 显示 pool_factor=3 时保留了 97.8% 的性能

### 9.3 LambdaTokenPooler —— 自定义池化

```python
# 文件: colpali_engine/compression/token_pooling/lambda_token_pooling.py

class LambdaTokenPooler(BaseTokenPooler):
    """
    使用用户自定义函数进行池化。
    """
    def __init__(self, pool_func: Callable[[torch.Tensor], torch.Tensor]):
        self.pool_func = pool_func
        # 例如: pool_func = lambda x: x[::2]  # 每隔一个token取一个

    def _pool_embeddings_impl(self, embeddings, **kwargs):
        pooled = [self.pool_func(emb) for emb in embeddings]
        return pooled, None
```

### 9.4 压缩效果

| pool_factor | 向量数量 | 减少比例 | 性能保留 |
|-------------|---------|---------|---------|
| 1 (无压缩) | ~1030 | 0% | 100% |
| 2 | ~515 | 50% | ~99% |
| 3 | ~343 | **66.7%** | **97.8%** |
| 5 | ~206 | 80% | ~95% |

每页存储：`向量数 × 128维 × 2字节(float16)` = 1030 × 128 × 2 = **263 KB** → 压缩后 **88 KB**。

---

## 10. 训练与评估实践指南

### 10.1 环境安装

```bash
# 克隆仓库
git clone https://github.com/illuin-tech/colpali.git
cd colpali

# 安装（含训练和可解释性依赖）
pip install -e ".[all]"
```

依赖版本约束（来自 `pyproject.toml`）：
- `transformers>=5.2.0,<6.0.0`
- `peft>=0.18.0,<0.19.0`
- `torch>=2.2.0,<2.11.0`
- `accelerate>=1.1.0,<2.0.0`（训练用）
- `einops>=0.8.0,<1.0.0`（可解释性用）
- `scipy`（token 压缩用）
- `configue>=5.0.0`（YAML 配置加载）

### 10.2 训练配置详解

训练使用 `configue` 库加载 YAML 配置文件。核心配置在 `scripts/configs/` 目录下。

**启动训练**:

```bash
# 文件: scripts/train/train_colbert.py
python scripts/train/train_colbert.py scripts/configs/pali/train_colpali_model.yaml
```

训练入口使用 `typer` CLI 框架：

```python
# 文件: scripts/train/train_colbert.py

@app.command()
def main(config_file: Path) -> None:
    config = configue.load(config_file, sub_path="config")
    # configue.load 解析 YAML 文件中的 () 标记，
    # 自动实例化对应的 Python 类

    if isinstance(config, ColModelTrainingConfig):
        training_app = ColModelTraining(config)

    if config.run_train:
        training_app.train()
        training_app.save()
```

**模型配置示例**（`scripts/configs/pali/train_colpali_model.yaml`）：

```yaml
config:
  (): colpali_engine.trainer.colmodel_training.ColModelTrainingConfig
  # configue 语法: () 指定要实例化的类

  output_dir: !path ../../../models/right_pad/train_colpali-3b-mix-448
  
  # === 处理器 ===
  processor:
    (): colpali_engine.utils.transformers_wrappers.AllPurposeWrapper
    class_to_instanciate: !ext colpali_engine.models.ColPaliProcessor
    pretrained_model_name_or_path: "./models/colpaligemma-3b-mix-448-base"
    max_length: 50
    # max_length=50: 查询最多 50 个 token
  
  # === 模型 ===
  model:
    (): colpali_engine.utils.transformers_wrappers.AllPurposeWrapper
    class_to_instanciate: !ext colpali_engine.models.ColPali
    pretrained_model_name_or_path: "./models/colpaligemma-3b-mix-448-base"
    torch_dtype: !ext torch.bfloat16
    # bfloat16 精度: 减少显存占用，同时保持训练稳定性
  
  # === 数据 ===
  train_dataset:
    (): colpali_engine.utils.dataset_transformation.load_train_set
  eval_dataset: !import ../data/test_data.yaml
  # !import: configue 的跨文件引用语法
  
  # === 损失函数 ===
  max_length: 50
  loss_func:
    (): colpali_engine.loss.late_interaction_losses.ColbertPairwiseCELoss
    # 使用 pairwise CE loss（带正负样本对比）
  
  # === LoRA 配置 ===
  peft_config:
    (): peft.LoraConfig
    r: 32            # LoRA 秩
    lora_alpha: 32   # 缩放因子 (alpha/r = 1.0)
    lora_dropout: 0.1
    init_lora_weights: "gaussian"
    bias: "none"
    task_type: "FEATURE_EXTRACTION"
    target_modules: '(.*(language_model).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$|.*(custom_text_proj).*$)'
    # 正则表达式选择要加 LoRA 的模块:
    # 1. language_model 中的所有 attention/FFN 投影层
    # 2. custom_text_proj（128维投影层）
```

**训练参数**（`scripts/configs/tr_args/default_tr_args.yaml`）：

```yaml
(): transformers.training_args.TrainingArguments
num_train_epochs: 1
per_device_train_batch_size: 4        # 每卡 batch size
per_device_eval_batch_size: 4
eval_strategy: "steps"
save_steps: 500
logging_steps: 10
eval_steps: 100
warmup_steps: 500
learning_rate: 5e-5                   # 学习率
save_total_limit: 1
```

> 论文: batch size = 32（4卡 × 8），学习率 5e-5，训练 1 epoch。

### 10.3 ViDoRe 评估数据集

评估配置文件（`scripts/configs/data/test_data.yaml`）指定了 ViDoRe 基准的数据子集：

| 数据集 | 领域 | 特点 |
|--------|------|------|
| DocVQA | 文档问答 | 扫描文档 |
| InfoVQA | 信息图表问答 | 复杂布局 |
| ArXivQA | 学术论文问答 | 数学公式、图表 |
| TabFQuAD | 法语表格问答 | 多语言 |
| TAT-DQA | 表格+文本问答 | 混合内容 |
| SHIFT Project | 气候报告 | 领域特定 |
| SyntheticDocQA | 合成问答 | 能源/医疗/AI/政府多领域 |

### 10.4 本仓库的评估流程

本仓库扩展了 ColPali，重点在**鲁棒性实验**。`experiments/run_benchmark.py` 实现了完整的评估流程：

```python
# 文件: experiments/run_benchmark.py

# ===== 使用 ColQwen2（性能优于原始 ColPali）=====
model = ColQwen2.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda").eval()
processor = ColQwen2Processor.from_pretrained(PROCESSOR_NAME)

# ===== 编码查询 =====
for batch in queries:
    inputs = processor.process_queries(batch).to("cuda")
    with torch.no_grad():
        q_vecs = model(**inputs)  # (batch, query_tokens, 128)

# ===== 编码文档（在此注入退化/恢复流程）=====
for batch in images:
    preprocessed = preprocess_fn(batch)  # ← 降质/修复处理
    inputs = processor.process_images(preprocessed).to("cuda")
    with torch.no_grad():
        d_vecs = model(**inputs)  # (batch, image_tokens, 128)

# ===== 计算相似度矩阵 =====
scores_matrix = processor.score_multi_vector(q_tensor, d_tensor)
# (n_queries, n_docs) — MaxSim 分数

# ===== 计算指标 =====
ndcg_at_k(scores, relevant_docs, k=5)
recall_at_k(ranked_list, relevant_docs, k=5)
mean_reciprocal_rank(ranked_list, relevant_docs)
```

**评估命令示例**:

```bash
# 干净条件（基准线）
python experiments/run_benchmark.py --condition clean

# 降质条件
python experiments/run_benchmark.py --condition degraded --deg heavy_noise

# 修复条件
python experiments/run_benchmark.py --condition restored --deg heavy_noise --rest nlmeans

# 分割条件
python experiments/run_benchmark.py --condition segmented
```

### 10.5 configue 库说明

本仓库大量使用 [`configue`](https://github.com/illuin-tech/configue) 配置库，它是仓库维护者 illuin-tech 开发的：

```yaml
# configue 特殊语法:

# 1. () — 指定要实例化的类
config:
  (): some_module.SomeClass    # → SomeClass(其他参数作为kwargs)

# 2. !ext — 引用外部 Python 对象（不实例化）
torch_dtype: !ext torch.bfloat16    # → torch.bfloat16

# 3. !import — 导入另一个 YAML 文件
eval_dataset: !import ../data/test_data.yaml   # → 加载并解析该文件

# 4. !path — 自动解析为 Path 对象
output_dir: !path ../../../models/xxx    # → Path("../../../models/xxx")
```

这使得整个训练管线可以通过纯 YAML 配置来指定模型、处理器、损失函数、数据集等所有组件。

---

## 11. 端到端流程总结

### 11.1 训练流程

```
┌────────────────────────────────────────────────────────────┐
│                     训练数据准备                             │
│  load_train_set() → ColPaliEngineDataset                   │
│  每条样本: {query, pos_target(page_image), neg_target?}     │
└─────────────────────────┬──────────────────────────────────┘
                          ↓
┌────────────────────────────────────────────────────────────┐
│                     数据整理 (Collator)                     │
│  VisualRetrieverCollator.__call__()                         │
│  queries  → [prefix + query + pad] → tokenize              │
│  images   → process_images → pixel_values                  │
│  neg_imgs → process_images → pixel_values                  │
└─────────────────────────┬──────────────────────────────────┘
                          ↓
┌────────────────────────────────────────────────────────────┐
│                     前向传播 (Forward)                       │
│  ColPali/ColQwen2.forward(input_ids, pixel_values, ...)    │
│  VLM backbone → (batch, seq_len, hidden_dim)               │
│  Linear projection → (batch, seq_len, 128)                 │
│  L2 normalize → 单位球面上的嵌入                             │
└─────────────────────────┬──────────────────────────────────┘
                          ↓
┌────────────────────────────────────────────────────────────┐
│                     损失计算 (Loss)                         │
│  1. DDP all_gather: 收集所有 GPU 的嵌入                     │
│  2. score_multi_vector(): einsum → MaxSim 分数矩阵          │
│  3. ColbertPairwiseCELoss: 正样本对角线 vs 负样本            │
│  4. 反向传播 + 优化器更新                                    │
└────────────────────────────────────────────────────────────┘
```

### 11.2 推理（检索）流程

```
┌─────────────────┐        ┌──────────────────┐
│   PDF 文档库     │        │    用户查询       │
│  (大量页面图像)   │        │ "What is the..." │
└────────┬────────┘        └────────┬─────────┘
         ↓                          ↓
┌────────────────┐         ┌────────────────┐
│ process_images │         │ process_queries │
│ → pixel_values │         │ → input_ids     │
└────────┬───────┘         └────────┬────────┘
         ↓                          ↓
┌────────────────┐         ┌────────────────┐
│ model.forward  │         │ model.forward  │
│ → (N_d, 128)   │         │ → (N_q, 128)   │
│ 多向量嵌入      │         │ 多向量嵌入      │
└────────┬───────┘         └────────┬────────┘
         ↓                          ↓
         └──────────┐    ┌──────────┘
                    ↓    ↓
            ┌──────────────────┐
            │   MaxSim 打分     │
            │ einsum → amax    │
            │ → sum → score    │
            └────────┬─────────┘
                     ↓
            ┌──────────────────┐
            │  排序 → Top-K    │
            │  返回最相关页面   │
            └──────────────────┘
```

### 11.3 关键设计决策

| 设计决策 | 选择 | 原因 |
|---------|------|------|
| 视觉编码器 | SigLIP (ColPali) / ViT (ColQwen2) | 原生图像理解，无需 OCR |
| 嵌入交互 | 延迟交互 (Late Interaction) | 平衡效率与精度 |
| 微调方法 | LoRA (r=32) | 仅训练 ~0.5% 参数 |
| 投影维度 | 128 | ColBERT 最佳实践 |
| 损失函数 | ColbertPairwiseCELoss | 原始论文默认 |
| 归一化 | L2 归一化 | MaxSim 等价于余弦相似度 |
| 训练精度 | bfloat16 | 减少显存 + 训练稳定 |
| Patch 策略 | 固定分辨率 (ColPali) / 动态分辨率 (ColQwen2) | 效率 vs 精度 tradeoff |

### 11.4 论文核心结果（Table 2）

ColPali 在 ViDoRe 基准上的表现 (nDCG@5):
- **ColPali**: 81.3（超越所有传统 OCR+检索方法）
- **ColQwen2-v0.1**: 86.6（+5.3）
- **对比**: BM25 on OCR = 68.5, BGE-M3 = 73.5, Jina-CLIP = 45.3

这证明了直接从页面图像进行视觉检索的有效性。

---

## 12. 仓库文件索引

| 文件路径 | 功能 |
|---------|------|
| `colpali_engine/models/paligemma/colpali/modeling_colpali.py` | ColPali 模型架构 |
| `colpali_engine/models/paligemma/colpali/processing_colpali.py` | ColPali 处理器 |
| `colpali_engine/models/paligemma/bipali/modeling_bipali.py` | BiPali 基线模型 |
| `colpali_engine/models/qwen2/colqwen2/modeling_colqwen2.py` | ColQwen2 模型架构 |
| `colpali_engine/models/qwen2/colqwen2/processing_colqwen2.py` | ColQwen2 处理器 |
| `colpali_engine/loss/late_interaction_losses.py` | 全部损失函数 |
| `colpali_engine/collators/visual_retriever_collator.py` | 数据批处理 |
| `colpali_engine/data/dataset.py` | 数据集类 |
| `colpali_engine/data/sampler.py` | 多数据集采样器 |
| `colpali_engine/trainer/colmodel_training.py` | HuggingFace 训练配置 |
| `colpali_engine/trainer/contrastive_trainer.py` | 对比学习训练器 |
| `colpali_engine/trainer/colmodel_torch_training.py` | 原生 PyTorch DDP 训练 |
| `colpali_engine/utils/processing_utils.py` | 基类、MaxSim、FastPlaid |
| `colpali_engine/interpretability/similarity_map_utils.py` | 相似度图计算 |
| `colpali_engine/interpretability/similarity_maps.py` | 热图可视化 |
| `colpali_engine/compression/token_pooling/base_token_pooling.py` | 池化基类 |
| `colpali_engine/compression/token_pooling/hierarchical_token_pooling.py` | 层次聚类池化 |
| `colpali_engine/compression/token_pooling/lambda_token_pooling.py` | 自定义池化 |
| `scripts/train/train_colbert.py` | 训练入口脚本 |
| `scripts/configs/pali/train_colpali_model.yaml` | ColPali 训练配置 |
| `experiments/run_benchmark.py` | 鲁棒性基准测试 |
| `experiments/config.py` | 实验配置 |
| `robust/degradation/pipeline.py` | 图像降质管线 |
| `robust/restoration/` | 图像修复模块 |
| `robust/segmentation/` | 文档分割模块 |
| `robust/evaluation/metrics.py` | 评估指标 (nDCG, Recall, MRR) |

