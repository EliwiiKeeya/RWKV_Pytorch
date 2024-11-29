import time
import os
import torch
from src.model import RWKV_RNN
from src.sampler import sample_logits
from src.rwkv_tokenizer import RWKV_TOKENIZER
if __name__ == '__main__':
    args = {
        'MODEL_NAME': 'weight/RWKV-x060-World-3B-v2.1-20240417-ctx4096', #模型文件的名字，pth结尾的权重文件。
        'vocab_size': 65536, #词表大小
        'device': "cuda", # 运行设备，可选'cpu','cuda','musa','npu'
        'onnx_opset': '12',
    }
    device = args['device']
    assert device in ['cpu','cuda','musa','npu']
    
    # 如果是国产硬件，需要 import 插件来 hack pytorch
    if device == "musa":
        import torch_musa
    elif device == "npu":
        import torch_npu
    
    # 加载模型和分词器
    print("Loading model and tokenizer...")
    model = RWKV_RNN(args).to(device)
    tokenizer = RWKV_TOKENIZER("asset/rwkv_vocab_v20230424.txt")
    print("Done.")
    
    # 设置续写的初始字符串和参数
    initial_string = "User: 帮我用python写一个打印字符三角形的代码.\n\nAssistant: "
    TEMPERATURE = 2.5  # 温度参数
    TOP_P = 0.1  # Top-p采样参数
    LENGTH_PER_TRIAL = 50  # 生成的长度
    
    # 编码初始字符串
    token = torch.LongTensor(tokenizer.encode(initial_string)).to(device)

    # 初始化状态
    state = torch.zeros(1, model.state_size[0], model.state_size[1]).to(device) 
    with torch.no_grad():
        for t in token.reshape(-1, 1):
            out, state = model.forward(t, state)
        else:
            token_sampled = sample_logits(out, TEMPERATURE, TOP_P)
            token = torch.cat((token, token_sampled.unsqueeze(1)), 1)

    start_time = time.time() # 开始计时        
    for step in range(LENGTH_PER_TRIAL):  # 生成指定数量的token
        # 使用GPU来完成采样工作，使得GPU有更高的利用率
        with torch.no_grad():
            out, state = model.forward(token_sampled, state)
        token_sampled = sample_logits(out, TEMPERATURE, TOP_P)
        token = torch.cat((token, token_sampled.unsqueeze(1)), 1)

        # 清除屏幕并打印结果
        os.system('cls' if os.name == 'nt' else 'clear')
        decoded_sequences = tokenizer.decode(token.cpu().tolist())
        for i, seq in enumerate(decoded_sequences):
            print(f"Batch {i+1}: {seq}")

    end_time = time.time() # 结束计时

    total_time = end_time - start_time
    tokens_generated = LENGTH_PER_TRIAL
    speed = tokens_generated / total_time
    print(f"\nTotal time: {total_time:.2f} seconds")
    print(f"Tokens generated: {tokens_generated}")
    print(f"Token generation speed: {speed:.2f} tokens/second")