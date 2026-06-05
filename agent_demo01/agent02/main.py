import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    # 指定模型ID
    model_id = "Qwen/Qwen1.5-0.5B-Chat"
    
    # 设置设备，优先使用GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 加载分词器
    print("正在加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # 加载模型，并将其移动到指定设备
    print("正在加载模型... (首次运行会下载模型文件，可能需要一些时间)")
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    
    print("模型和分词器加载完成！\n")
    
    # 准备对话输入
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "你好，请介绍你自己。"}
    ]
    
    # 使用分词器的模板格式化输入
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # 编码输入文本
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    
    print("编码后的输入文本:")
    print(model_inputs)
    
    # 使用模型生成回答
    print("\n正在生成回答...")
    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512
    )
    
    # 将生成的 Token ID 截取掉输入部分
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    # 解码生成的 Token ID
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    print("\n模型的回答:")
    print(response)

if __name__ == "__main__":
    main()