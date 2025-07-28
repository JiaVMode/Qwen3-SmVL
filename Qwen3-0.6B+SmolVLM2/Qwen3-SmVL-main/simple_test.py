#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的模型测试脚本
使用与训练脚本相同的方式加载和测试模型
"""

import os
import sys
import torch
from PIL import Image

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import load_processor, load_model


def test_model_with_image(image_path, device="cuda:0"):
    """使用与训练脚本相同的方式测试模型"""
    print("🚀 开始测试模型...")
    print(f"图像路径: {image_path}")
    print(f"使用设备: {device}")
    
    try:
        # 加载处理器和模型（与训练时相同的方式）
        print("📥 加载处理器...")
        processor = load_processor()
        print("✅ 处理器加载成功")
        
        print("📥 加载模型...")
        model = load_model(device)
        
        # 尝试加载训练好的权重
        model_path = "/home/lab/LJ/play/Qwen3-0.6B+SmolVLM2/Qwen3-SmVL-main/model/freeze_llm_vlm_cocoqa"
        if os.path.exists(os.path.join(model_path, "model.safetensors")):
            print("📥 加载训练好的权重...")
            from safetensors import safe_open
            checkpoint_path = os.path.join(model_path, "model.safetensors")
            with safe_open(checkpoint_path, framework="pt", device=device) as f:
                state_dict = {key: f.get_tensor(key) for key in f.keys()}
            
            # 加载权重，允许部分不匹配
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            if missing_keys:
                print(f"⚠️  缺失的键: {missing_keys}")
            if unexpected_keys:
                print(f"⚠️  意外的键: {unexpected_keys}")
            print("✅ 训练好的权重加载成功")
        
        model.eval()
        print("✅ 模型加载成功")
        
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        print(f"✅ 图像加载成功，尺寸: {image.size}")
        
        # 构建测试问题
        question = "请详细描述这张图片中的内容。"
        print(f"📝 问题: {question}")
        
        # 构建消息（与训练脚本相同的方式）
        messages = [
            {
                "role": "system",
                "content": "使用中文回答所有问题。",
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            },
        ]
        
        # 应用聊天模板
        texts = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
        print("################# 输入文本 #################")
        print(texts)
        
        # 构建批次（与训练脚本相同的方式）
        images = [[image]]
        batch = processor(
            text=[texts],
            images=images,
            max_length=1024,
            return_tensors="pt",
            padding_side="left",
            padding=True,
        ).to(model.device, dtype=torch.bfloat16)
        
        # 生成响应
        print("🧪 开始生成...")
        with torch.no_grad():
            generated_ids = model.generate(
                **batch, 
                do_sample=False, 
                max_new_tokens=256
            )
        
        # 解码输出
        model_context = processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )
        input_ids_len = batch["input_ids"].shape[1]
        generated_texts = processor.batch_decode(
            generated_ids[:, input_ids_len:], skip_special_tokens=True
        )
        
        print("################# 生成文本 #################")
        print(generated_texts[0])
        print("✅ 测试完成！")
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="简化模型测试")
    parser.add_argument("--image", type=str, required=True,
                       help="测试图像路径")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="设备类型 (cuda:0, cpu)")
    
    args = parser.parse_args()
    
    # 检查CUDA可用性
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("⚠️  CUDA不可用，切换到CPU")
        args.device = "cpu"
    
    test_model_with_image(args.image, args.device) 