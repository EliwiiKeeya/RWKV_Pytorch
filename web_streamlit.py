import torch
import streamlit as st
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

# 生成文本的生成器函数
def generate_text_stream(prompt: str, temperature=1.5, top_p=0.1, max_tokens=2048, stop=['\n\nUser']):
    encoded_input = tokenizer.encode([prompt])
    token = torch.tensor(encoded_input).long().to(device)
    state = torch.zeros(1, model.state_size[0], model.state_size[1]).to(device)
    prompt_tokens = len(encoded_input[0])

    with torch.no_grad():
        token_out, state_out = model.forward_parallel(token, state)
        
    del token
    
    out = token_out[:, -1]
    generated_tokens = ''
    completion_tokens = 0

    for _ in range(max_tokens):
        token_sampled = sample_logits(out, temperature, top_p)
        with torch.no_grad():
            out, state = model.forward(token_sampled, state)
        
        last_token = tokenizer.decode(token_sampled.unsqueeze(1).tolist())[0]
        generated_tokens += last_token
        completion_tokens += 1
        
        if generated_tokens.endswith(tuple(stop)):
            break
        else:
            yield f"{last_token}"
    yield "data: [DONE]"


st.title("Simple chat")
model, tokenizer, device = init_model()
prompt = str()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
    prompt += f"{message['role']}: {message['content']}\n\n"
    
        
# Streamed response emulator
def response_generator():
    global prompt
    if prompt:
        word = str()            
        response = generate_text_stream(prompt)
        while True:
            delta = next(response)
            if delta == "data: [DONE]":
                break
            yield word + delta

# Accept user input
user_input = st.chat_input("What is up?")
if user_input:
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(user_input)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    prompt += "User: " + str(user_input) + "\n\nAssistant: "

    with st.chat_message("assistant"):
        response = st.write_stream(response_generator())
        # Add assistant response to chat history
        response = response.strip()
        st.session_state.messages.append({"role": "assistant", "content": str(response)})