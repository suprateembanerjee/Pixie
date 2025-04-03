import streamlit as st
import ollama
import time
import json

from prompts import system_prompt_ea, system_prompt_ca, system_prompt_poet

def chat_with_model(model):
    """Sends user input to the selected Ollama model and returns the response."""
    start_time = time.time()
    response = ollama.chat(model=model, messages=st.session_state.messages[model])
    end_time = time.time()
    elapsed_time = end_time - start_time
    return response['message']['content'], elapsed_time

# Streamlit UI
st.set_page_config(page_title="Ollama Chat", layout="wide")
st.title("Ollama Chat")

# Sidebar to select model
models = [model['model'] for model in ollama.list()['models']]
selected_model = st.sidebar.selectbox("Choose a model", models, index=0)

if 'system_prompt' not in st.session_state:
    st.session_state.system_prompt = system_prompt_ea

st.sidebar.text_area(label='System Prompt', height=200, key='system_prompt_area', value=st.session_state.system_prompt)


if st.sidebar.button('Executive Assistant'):
    st.session_state.system_prompt = system_prompt_ea
    st.session_state.messages[selected_model][0] = {"role": "system", "content": st.session_state.system_prompt}
    st.rerun()

if st.sidebar.button('Coding Assistant'):
    st.session_state.system_prompt = system_prompt_ca
    st.session_state.messages[selected_model][0] = {"role": "system", "content": st.session_state.system_prompt}
    st.rerun()

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = {}
    for model in models:
        st.session_state.messages[model] = [{"role": "system", "content": system_prompt_ea}]

# Display chat history
for message in st.session_state.messages[selected_model][1:]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
user_input = st.chat_input("Type your message...")
if user_input:
    # Append user message to chat history
    if st.session_state.messages[selected_model][0] != st.session_state.system_prompt_area:
        st.session_state.messages[selected_model][0] = {"role": "system", "content": st.session_state.system_prompt_area}
    st.session_state.messages[selected_model].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Get response from model
    response, elapsed_time = chat_with_model(selected_model)
    
    # Append model response to chat history
    st.session_state.messages[selected_model].append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
        st.caption(f"⏱️ Response time: {elapsed_time:.2f} seconds")