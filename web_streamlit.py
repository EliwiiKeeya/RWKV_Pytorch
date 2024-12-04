﻿import torch
import streamlit as st
from audio import SVCAudioWebInterface
from src.model import RWKV_RNN
from src.sampler import sample_logits
from src.rwkv_tokenizer import RWKV_TOKENIZER


@st.cache_resource
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


class App(TextGenerator):
    def __init__(self) -> None:
        super().__init__()
        self.text_filter = TextFilter()
        self.audio_interface = SVCAudioWebInterface()

    def __call__(self):
        if self.prompt:
            self.text_filter.content = ""
            response = self._generate_text_stream()
            while True:
                delta = next(response)
                if delta == "data: [DONE]":
                    self.text_filter.content = self.text_filter.content.strip()
                    if self.text_filter.content:
                        print(self.text_filter.content)
                        self.audio_interface.process(self.text_filter.content)
                    break
                yield delta
                sentence = self.text_filter(delta)
                if sentence:
                    print(sentence)
                    self.audio_interface.process(sentence)


st.title("Simple chat")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "text_generator" not in st.session_state:
    st.session_state.text_generator = App()

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
    st.session_state.text_generator.prompt = f"{message['role']}: {message['content']}\n\n"

# Accept user input
user_input = st.chat_input("What is up?")
if user_input:
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(user_input)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.text_generator.prompt = "User: " + str(user_input) + "\n\nAssistant: "

    with st.chat_message("assistant"):
        response = st.write_stream(st.session_state.text_generator())
        # Add assistant response to chat history
        response = response.strip()
        st.session_state.messages.append({"role": "assistant", "content": str(response)})
