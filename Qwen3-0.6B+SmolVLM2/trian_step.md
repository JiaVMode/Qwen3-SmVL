# 环境配置
```bash
conda create -n Qwen3-SmVL python=3.11 -y
conda activate Qwen3-SmVL
```

## 安装依赖库
```bash
pip install -r requirements.txt
```

# 数据集和模型下载
```bash
bash download_resource.sh
```

# 微调训练
```bash
# 单GPU训练
torchrun train.py ./cocoqa_train.yaml
# 4GPU训练
torchrun --nproc_per_node=2 train.py ./cocoqa_train.yaml
```

# 测试
```bash
python simple_test.py --image path-to-image
# 例如
python simple_test.py --image ./resource/cocoqa_swanlab.png
````