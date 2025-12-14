#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Qwen2.5-0.5B-Instruct LoRA/QLoRA 微调脚本
该脚本用于对 Qwen2.5-0.5B-Instruct 模型进行低秩适应(LoRA)或量化低秩适应(QLoRA)微调
"""

import torch
from peft import TaskType, LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
import pandas as pd
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    DataCollatorForSeq2Seq, TrainingArguments, 
    Trainer, GenerationConfig
)

# 定义要微调的基础模型名称
TRAINING_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

def load_dataset_json(path):
    """
    加载JSON格式的数据集
    
    :param path: 数据集文件的完整路径
    :return: 转换为Hugging Face Dataset格式的数据集
    """
    # 使用pandas读取JSON文件
    df = pd.read_json(path)
    # 将pandas DataFrame转换为Hugging Face Dataset
    ds = Dataset.from_pandas(df)
    return ds


def dataset_preprocess(ds):
    """
    对数据集进行预处理，包括加载分词器和处理数据样本
    
    :param ds: Hugging Face Dataset格式的原始数据集
    :return: 预处理后的数据集和使用的分词器
    """
    # 加载预训练分词器
    # use_fast=False: 使用慢速分词器，支持更复杂的文本处理
    # trust_remote_code=True: 信任模型提供的自定义代码
    tokenizer = AutoTokenizer.from_pretrained(TRAINING_MODEL, use_fast=False, trust_remote_code=True)
    
    def process_func(example):
        """
        处理单个数据样本的内部函数
        
        :param example: 单个数据样本，包含instruct、input和output字段
        :return: 处理后的样本，包含input_ids、attention_mask和labels
        """
        MAX_LENGTH = 384    # 最大序列长度限制
        input_ids, attention_mask, labels = [], [], []
        
        # 构建指令部分的输入，使用模型要求的对话格式
        instruction = tokenizer(
            f"<|im_start|>system\n现在你要扮演会议语音助手<|im_end|>\n<|im_start|>user\n{example['instruct'] + example['input']}<|im_end|>\n<|im_start|>assistant\n", 
            add_special_tokens=False  # 不自动添加特殊标记，因为我们已经手动添加
        )
        
        # 构建响应部分的输入
        response = tokenizer(f"{example['output']}", add_special_tokens=False)
        
        # 合并指令和响应的token ids，并添加pad_token作为结束
        input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
        # 合并注意力掩码，pad_token位置设置为1表示需要关注
        attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
        # 构建标签：指令部分用-100屏蔽（不参与损失计算），响应部分保留实际token id
        labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]  
        
        # 截断超过最大长度的序列
        if len(input_ids) > MAX_LENGTH:
            input_ids = input_ids[:MAX_LENGTH]
            attention_mask = attention_mask[:MAX_LENGTH]
            labels = labels[:MAX_LENGTH]
            
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
    
    # 应用处理函数到整个数据集，并移除原始列名
    tokenized_id = ds.map(process_func, remove_columns=ds.column_names)
    
    # 解码并打印第一个样本的输入，用于调试
    tokenizer.decode(tokenized_id[0]['input_ids'])
    # 解码并打印第二个样本的标签（过滤掉-100），用于调试
    tokenizer.decode(list(filter(lambda x: x != -100, tokenized_id[1]["labels"])))

    return tokenized_id, tokenizer


def train(tokenized_id, tokenizer, cpu=True):
    """
    执行模型微调的主函数
    
    :param tokenized_id: 预处理后的数据集
    :param tokenizer: 分词器
    :param cpu: 是否使用CPU进行训练，默认为True（CPU训练）
    """
    # 根据是否使用CPU选择数据类型
    # CPU训练使用bfloat16，GPU训练使用float16
    dtype = torch.bfloat16 if cpu else torch.float16

    # 加载预训练模型
    # device_map="auto": 自动分配模型到可用设备（CPU或GPU）
    # torch_dtype=dtype: 设置模型的数据类型
    model = AutoModelForCausalLM.from_pretrained(TRAINING_MODEL, device_map="auto", torch_dtype=dtype)
    
    # 启用输入梯度检查点，减少内存使用
    model.enable_input_require_grads()

    # 配置LoRA参数
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,  # 任务类型为因果语言模型
        # 指定要微调的模型模块
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=False,  # 训练模式（True为推理模式）
        r=8,  # LoRA秩，控制适配器的维度
        lora_alpha=32,  # LoRA缩放因子，通常为r的4倍
        lora_dropout=0.1  # Dropout比例，防止过拟合
    )
    
    # 应用LoRA配置到模型
    model = get_peft_model(model, config)

    # 配置训练参数
    args = TrainingArguments(
        output_dir=f"./output/{TRAINING_MODEL}_lora",  # 模型输出目录
        per_device_train_batch_size=4,  # 每个设备的训练批量大小
        gradient_accumulation_steps=4,  # 梯度累积步数，实际批量大小=4*4=16
        logging_steps=10,  # 每10步记录一次日志
        num_train_epochs=8,  # 训练轮数
        save_steps=100,  # 每100步保存一次模型
        learning_rate=1e-4,  # 学习率
        save_on_each_node=True,  # 在每个节点上保存模型
        gradient_checkpointing=True,  # 启用梯度检查点，减少内存使用
    )  

    # 创建训练器
    trainer = Trainer(
        model=model,  # 要训练的模型
        args=args,  # 训练参数
        train_dataset=tokenized_id,  # 训练数据集
        # 数据整理器，用于处理批量数据
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )  
    
    # 开始训练
    trainer.train() 


def combine_and_save_models(cpu=True):
    """
    合并LoRA适配器和基础模型，并保存合并后的模型
    
    :param cpu: 是否使用CPU进行模型合并，默认为True
    """
    # 导入必要的库（函数内部导入，避免不必要的依赖加载）
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    from peft import PeftModel

    # 根据是否使用CPU选择数据类型
    dtype = torch.bfloat16 if cpu else torch.float16
    model_path = TRAINING_MODEL
    # LoRA检查点路径（假设是最后一个保存的检查点）
    lora_path = f'./output/{TRAINING_MODEL}_lora/checkpoint-100' 

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # 加载预训练模型（评估模式）
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        device_map="auto",  # 自动分配设备
        torch_dtype=dtype,  # 设置数据类型
        trust_remote_code=True  # 信任自定义代码
    ).eval()  # 设置为评估模式

    # 加载LoRA适配器并合并到基础模型
    model = PeftModel.from_pretrained(model, model_id=lora_path)
    model = model.merge_and_unload()  # 合并权重并卸载peft封装

    # 保存合并后的模型
    merged_model_path = f'./merged_{TRAINING_MODEL}_lora'
    model.save_pretrained(merged_model_path)
    tokenizer.save_pretrained(merged_model_path)
    print(f"Merged model saved to {merged_model_path}")


# 主程序入口
if __name__ == "__main__":
    print("Start to train...")
    # 训练数据集路径
    training_dataset = "./data/train.json"

    # 加载数据集
    ds = load_dataset_json(training_dataset)
    print(f'dataset loaded, train size: {len(ds)}, {ds[0:3]}')

    # 预处理数据集
    tokenized_id, tokenizer = dataset_preprocess(ds)
    print(f'dataset preprocessed, start to run training...')

    # 执行训练
    train(tokenized_id, tokenizer, cpu=True)

    # 合并并保存模型
    print("Start to combine and save models...")
    combine_and_save_models(cpu=True)

    print("Done")