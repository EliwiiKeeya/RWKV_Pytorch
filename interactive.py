import torch
from audio import SVCAudioWebInterface
from src.model import RWKV_RNN
from src.sampler import sample_logits
from src.rwkv_tokenizer import RWKV_TOKENIZER


def init_model():
    # 模型参数配置
    args = {
        'MODEL_NAME': 'E:/Resources/Models/rwkv-6-world/RWKV-x060-World-1B6-v2.1-20240328-ctx4096',
        'vocab_size': 65536,
        'device': "cuda",
        'onnx_opset': '18',
    }
    device = args['device']
    assert device in ['cpu', 'cuda']

    print("Loading model and tokenizer...")
    model = RWKV_RNN(args).to(device)
    tokenizer = RWKV_TOKENIZER("asset/rwkv_vocab_v20230424.txt")
    print("Done")
    print(f"Model name: {args.get('MODEL_NAME').split('/')[-1]}")
    return model, tokenizer, device


class TextGenerator:
    def __init__(self) -> None:
        self.model, self.tokenizer, self.device = init_model()
        self.prompt = "Hello World!"
        self.state = torch.zeros(1, self.model.state_size[0], self.model.state_size[1]).to(self.device) 

    def _generate_text_stream(self, temperature=1.5, top_p=0.1, max_tokens=2048, stop='\n\nUser'):
        # 编码初始字符串
        token = torch.LongTensor(self.tokenizer.encode(self.prompt)).to(self.device)

        # 初始化状态
        with torch.no_grad():
            for t in token.reshape(-1, 1):
                out, self.state = self.model.forward(t, self.state)
            else:
                token_sampled = sample_logits(out, temperature, top_p)

                last_token = self.tokenizer.decode(token_sampled.unsqueeze(1).tolist())[0]
                yield last_token
                # token = torch.cat((token, token_sampled.unsqueeze(1)), 1)
        
        word = str()
        for _ in range(max_tokens):  # 生成指定数量的token
            # 使用GPU来完成采样工作，使得GPU有更高的利用率
            with torch.no_grad():
                out, self.state = self.model.forward(token_sampled, self.state)
            token_sampled = sample_logits(out, temperature, top_p)
            
            try:
                last_token = self.tokenizer.decode(token_sampled.unsqueeze(1).tolist())[0]
                word += last_token
                if word.endswith(stop):
                    break
                else:
                    yield f"{last_token}"
            except:
                break
        yield "data: [DONE]"

    # 生成文本的生成器函数
    def __call__(self):
        if self.prompt:
            response = self._generate_text_stream()
            while True:
                delta = next(response)
                if delta == "data: [DONE]":    
                    break
                yield delta
                

class TextFilter:
    def __init__(self, s="") -> None:
        self.content = s

    def append(self, s: str):
        self.content += s

    def _main(self):
        symbols = "，。？！?!"
        for symbol in symbols:
            # 查找符号在字符串中的位置
            index = self.content.find(symbol)
            if index != -1:
                output = self.content[:index]
                self.content = self.content[index+1:]
                return output.strip()
        else:
            return ""

    def __call__(self, s: str=""):
        self.append(s)
        return self._main()

def main():
    text_generator = TextGenerator()
    text_filter = TextFilter()
    audio_interface = SVCAudioWebInterface()
    
    while True:
        print("User: ", end="")
        user_input = input()
        print("\nAssistant: ", end="")

        text_generator.prompt = "User: " + str(user_input) + "\n\nAssistant: "
        gen = text_generator()
        text_filter.content = ""
        for token in gen:
            print(token, end='')
            sentence = text_filter(token)
            if sentence:
                audio_interface.process(sentence)
                pass
        text_filter.content = text_filter.content.strip()
        if text_filter.content:
            audio_interface.process(text_filter.content)
            pass


if __name__ == "__main__":
    main()
