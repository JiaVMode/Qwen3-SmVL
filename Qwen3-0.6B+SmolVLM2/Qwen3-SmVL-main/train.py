import os
import sys
from dataclasses import dataclass
from functools import partial
from PIL import Image

import torch
from transformers import (
    TrainingArguments,
    Trainer,
    HfArgumentParser,
)
from transformers.trainer_utils import get_last_checkpoint
import datasets
import swanlab

from utils import load_model, load_processor

device = "cuda:0"

################
# 加载数据集
################
def load_mm_data(select_data):
    all_data_names = [
        "chartqa",
        "finqa",
        "aokvqa",
        # "mimic_cgd",  # bad dataset
        "figureqa",
        "diagram_image_to_text",
        "geomverse",
        "ai2d",
        "iam",
        "infographic_vqa",
        # "localized_narratives",  # bad dataset
        "intergps",
        "hateful_memes",
        "clevr",
        "iconqa",
        "multihiertt",
        "mapqa",
        "datikz",
        # "okvqa", # bad dataset
        "hitab",
        "chart2text",
        # "ocrvqa",  # bad dataset
        # "clevr_math", # bad dataset
        # "nlvr2",  # bad dataset
        "cocoqa",
        "docvqa",
        "dvqa",
    ]
    if select_data == "all":
        select_data = all_data_names
    elif select_data in all_data_names:
        select_data = [select_data]
    else:
        raise f"cannot find {select_data}"

    data_list = []
    for data_name in select_data:
        try:
            data_list.append(
                datasets.load_dataset("data/the_cauldron", data_name)["train"]
            )
        except:
            print(f"bad dataset:{data_name}")
    raw_data = datasets.concatenate_datasets(data_list)
    raw_data = raw_data.train_test_split(
        64, shuffle=True, seed=training_args.data_seed
    )  # 预留64条用于训练中测试，仅仅使用64条是因为减少测试时间长度
    if select_data == "all":
        raw_data["train"] = raw_data["train"].select(range(60 * 1024))  # 选取120M token
    return raw_data


################
# 冻结模型参数&打印模型可训练参数数
################
def freeze_model(qwen_smvl):
    for _, param in qwen_smvl.model.text_model.named_parameters():
        param.requires_grad = False
    for _, param in qwen_smvl.model.vision_model.named_parameters():
        param.requires_grad = False
    # for _, param in qwen_smvl.lm_head.named_parameters():
    #     param.requires_grad = False
    return qwen_smvl


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params/(2**20):.2f}M || all params: {all_param/(2**20):.2f}M || trainable%: {100 * trainable_params / all_param}"
    )


################
# 数据处理
################
def data_collate_fix2k(examples, processor, device, max_length=2048):
    batch_text = []
    batch_image = []
    for example in examples:
        images = example["images"][:1]  # 只允许一张图，不然显存顶不住
        batch_image.append(images)
        image_num = len(images)
        chat_texts = example["texts"][0]
        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}] * image_num
                + [{"type": "text", "text": chat_texts["user"]}],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": chat_texts["assistant"]}],
            },
        ]
        text = processor.apply_chat_template(
            messages, enable_thinking=False, add_generation_prompt=False
        )

        batch_text.append(text)

    batch = processor(
        text=batch_text,
        images=batch_image,
        max_length=max_length,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
    )
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    labels[labels == processor.image_token_id] = -100
    batch["labels"] = labels
    return batch.to(device, dtype=torch.bfloat16)


################
# 初始化训练参数
################
@dataclass
class MyTrainArgs(TrainingArguments):
    train_data: str = "cocoqa"  # 训练数据集名称
    seed: int = 42  # 随机种子
    data_seed: int = 42  # 数据随机种子
    max_steps: int = 200  # 最大训练步数
    per_device_train_batch_size: int = 1  # 每个设备的训练批次大小
    gradient_accumulation_steps: int = 4  # 梯度累积步数
    dataloader_pin_memory: bool = False  # 数据加载器是否固定内存
    warmup_ratio: float = 0.1  # 预热比例
    learning_rate: float = 1e-4  # 学习率
    lr_scheduler_type: str = "cosine"  # 学习率调度器类型
    weight_decay: float = 0.01  # 权重衰减
    logging_steps: int = 5  # 日志记录步数
    eval_strategy: str = "steps"  # 评估策略
    eval_steps: float = 0.125  # 评估步数
    save_strategy: str = "steps"  # 保存策略
    save_steps: float = 0.125  # 保存步数
    save_total_limit: int = 8  # 保存总数限制
    optim: str = "adamw_torch"  # 优化器
    bf16: bool = True  # 是否使用bf16
    output_dir: str = "./model/qwen-smovlm"  # 输出目录
    overwrite_output_dir: bool = True  # 是否覆盖输出目录
    report_to: str = "swanlab"  # 报告工具
    run_name: str = "freeze_except_connector_fulldata"  # 运行名称
    remove_unused_columns: bool = False  # 是否移除未使用的列
    gradient_checkpointing: bool = False  # 是否使用梯度检查点


def main(training_args):
    ################
    # 初始化模型&Tokenizer
    ################
    qwen_smvl_processor = load_processor()
    qwen_smvl = load_model(device)
    # 冻结参数
    qwen_smvl = freeze_model(qwen_smvl)
    # 打印可训练参数量
    print_trainable_parameters(qwen_smvl)

    ################
    # 准备训练数据集
    ################
    raw_data = load_mm_data(select_data=training_args.train_data)
    print(f"总数据条数：{raw_data}")

    # data formatting
    collate_fn = partial(
        data_collate_fix2k, processor=qwen_smvl_processor, device=device
    )

    ################
    # 开启训练
    ################
    last_checkpoint = None  # load last checkpoint if available
    if (
        os.path.isdir(training_args.output_dir)
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        print(
            f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
            "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
        )
    # Init Trainer
    trainer = Trainer(
        model=qwen_smvl,
        args=training_args,
        train_dataset=raw_data["train"],
        eval_dataset=raw_data["test"],
        data_collator=collate_fn,
    )
    trainer.train(resume_from_checkpoint=last_checkpoint)
    qwen_smvl.save_pretrained(training_args.output_dir)

    ################
    # 对单条数据进行推理
    ################
    with torch.no_grad():
        if trainer.state.is_world_process_zero:
            question = "图中有什么动物？"
            messages = [
                {
                    "role": "system",
                    "content": "使用中文回答所有问题。",
                    # "content": "使用中文回答所有问题，在<think>和</think>中写出思考过程，如果没有思考则为<think> </think>",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": question},
                    ],
                },
            ]
            texts = qwen_smvl_processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=True,
            )
            print("################# 输入文本 #################")
            print(texts)
            images = [[Image.open("dog.png")]]
            batch = qwen_smvl_processor(
                text=[texts],
                images=images,
                max_length=1024,
                return_tensors="pt",
                paddding_side="left",
                padding=True,
            ).to(qwen_smvl.device, dtype=torch.bfloat16)
            generated_ids = qwen_smvl.generate(
                **batch, do_sample=False, max_new_tokens=256
            )
            model_context = qwen_smvl_processor.batch_decode(
                generated_ids, skip_special_tokens=False
            )
            input_ids_len = batch["input_ids"].shape[1]
            generated_texts = qwen_smvl_processor.batch_decode(
                generated_ids[:, input_ids_len:], skip_special_tokens=True
            )
            print("################# 生成文本 #################")
            print(generated_texts[0])

            table = swanlab.echarts.Table()
            headers = ["输入问题", "模型输出"]
            rows = [[question, generated_texts[0]]]
            table.add(headers, rows)
            swanlab.log(
                {
                    "sample/输入图像": swanlab.Image(images[0][0]),
                    "sample/问题&回复": table,
                    "sample/上下文": swanlab.Text(model_context[0]),
                }
            )


if __name__ == "__main__":
    parser = HfArgumentParser(MyTrainArgs)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        # 如果只传递一个参数且是yaml文件路径，则解析它来获取参数
        # If we pass only one argument to the script and it's the path to a yaml file,
        # let's parse it to get our arguments.
        training_args = parser.parse_yaml_file(yaml_file=os.path.abspath(sys.argv[1]))[0]
    else:
        training_args = parser.parse_args_into_dataclasses()[0]
    main(training_args)
