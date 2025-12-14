from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel
import argparse  # 添加argparse模块用于命令行参数解析

model_path = 'Qwen/Qwen2.5-0.5B-Instruct'
lora_path = './output/Qwen/Qwen2.5-0.5B-Instruct_lora/checkpoint-100'


def test(instruct, cpu=True):
    """
    测试微调后的LoRA模型
    
    :param instruct: 用户指令内容，通过命令行传入
    :param cpu: 是否使用CPU进行推理，默认为True
    """
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # 根据设备选择数据类型和设备映射
    if cpu:
        dtype = torch.float16
        device_map = "cpu"  # 强制使用CPU
        device = "cpu"
    else:
        dtype = torch.bfloat16
        device_map = "auto"  # 自动分配设备
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # 加载基础模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        device_map=device_map,  # 使用明确的设备映射
        dtype=dtype, 
        trust_remote_code=True
    ).eval()  # 设置为评估模式

    # 加载LoRA权重
    model = PeftModel.from_pretrained(model, model_id=lora_path)

    # 测试对话
    inputs = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": "你是一个智能会议语音助手，请根据用户指令输出正确的指令和参数"},
            {"role": "user", "content": instruct}
        ],
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True
    )
    
    # 将输入移动到与模型相同的设备
    inputs = inputs.to(device)

    # 生成配置
    gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}
    
    # 生成回复
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        # 提取生成的部分
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        # 解码并打印结果
        print(tokenizer.decode(outputs[0], skip_special_tokens=True))


def parse_args():
    """
    解析命令行参数
    :return: 解析后的参数对象
    """
    parser = argparse.ArgumentParser(description="测试LoRA微调后的Qwen2.5-0.5B-Instruct模型")
    # 必选参数：用户指令
    parser.add_argument("--instruct", type=str, required=True, help="用户指令内容，例如：'打开麦克风'")
    # 可选参数：是否使用CPU，默认使用CPU
    parser.add_argument("--cpu", action="store_true", help="是否使用CPU进行推理（默认使用CPU）")
    # 可选参数：是否使用GPU
    parser.add_argument("--gpu", action="store_true", help="是否使用GPU进行推理（优先级高于--cpu）")
    return parser.parse_args()


if __name__ == "__main__":
    # 解析命令行参数
    args = parse_args()
    
    # 确定是否使用CPU
    # 如果同时指定了--cpu和--gpu，优先使用GPU
    use_cpu = not args.gpu and args.cpu
    
    # 调用测试函数
    test(instruct=args.instruct, cpu=use_cpu)