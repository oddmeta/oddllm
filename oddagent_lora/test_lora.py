from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel

model_path = 'Qwen/Qwen2.5-0.5B-Instruct'
lora_path = './output/Qwen/Qwen2.5-0.5B-Instruct_lora/checkpoint-100'


def test(cpu=True):
    """
    测试微调后的LoRA模型
    
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
        torch_dtype=dtype, 
        trust_remote_code=True
    ).eval()  # 设置为评估模式

    # 加载LoRA权重
    model = PeftModel.from_pretrained(model, model_id=lora_path)

    # 测试对话
    prompt = "你是谁？"
    inputs = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": "假设你是皇帝身边的女人--甄嬛。"},
            {"role": "user", "content": prompt}
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


if __name__ == "__main__":
    test()