"""
ColQwen2-v1.0 本地模型加载与推理测试脚本
使用本地下载好的 colqwen2-v1.0 文件夹进行测试
"""

import torch
from PIL import Image
from colpali_engine.models import ColQwen2, ColQwen2Processor

LOCAL_MODEL_PATH = "./colqwen2-v1.0"

def main():
    print("=" * 60)
    print("ColQwen2-v1.0 本地模型测试")
    print("=" * 60)

    # 1. 加载模型
    print("\n[1/4] 加载模型...")
    model = ColQwen2.from_pretrained(
        LOCAL_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    ).eval()
    print(f"  模型加载成功，设备: {model.device}")

    # 2. 加载处理器
    print("\n[2/4] 加载处理器...")
    processor = ColQwen2Processor.from_pretrained(LOCAL_MODEL_PATH)
    print("  处理器加载成功")

    # 3. 准备测试数据
    print("\n[3/4] 准备测试数据...")
    images = [
        Image.new("RGB", (128, 128), color="white"),
        Image.new("RGB", (64, 32), color="black"),
    ]
    queries = [
        "Is attention really all you need?",
        "What is the amount of bananas farmed in Salvador?",
    ]
    print(f"  测试图片数: {len(images)}, 测试查询数: {len(queries)}")

    # 4. 推理
    print("\n[4/4] 执行推理...")
    batch_images = processor.process_images(images).to(model.device)
    batch_queries = processor.process_queries(queries).to(model.device)

    with torch.no_grad():
        image_embeddings = model(**batch_images)
        query_embeddings = model(**batch_queries)

    scores = processor.score_multi_vector(query_embeddings, image_embeddings)
    print(f"  相似度分数矩阵:\n{scores}")

    print("\n" + "=" * 60)
    print("测试通过！ColQwen2-v1.0 本地模型加载与推理正常。")
    print("=" * 60)


if __name__ == "__main__":
    main()
