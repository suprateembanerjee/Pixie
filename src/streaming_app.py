import streamlit as st
from streamlit import session_state as ss
import ollama
import time
import re
from PIL import Image, ImageOps
from io import BytesIO
import pandas as pd

from prompts import system_prompt_ea, system_prompt_ca
from utils import get_models, get_tags, scrape_ollama_model
from enums import ModelType

st.set_page_config(page_title='Pixie', layout='wide')

# Styling
st.markdown(
            """
            <style>
            .my-title-font {
                font-family: 'Arial', sans-serif;
                font-size: 50px;
                font-weight: bold;
                color: white;
            }
            .chat {
                font-family: 'Courier New', monospace;
                font-size: 16px;
                color: forestgreen;
            }
            .reason {
                font-family: 'Courier New', monospace;
                font-size: 16px;
                color: royalblue;
            }
            .embed {
                font-family: 'Courier New', monospace;
                font-size: 16px;
                color: magenta;
            }
            .vision {
                font-family: 'Courier New', monospace;
                font-size: 16px;
                color: gold;
            }
            .author {
                font-family: 'Courier New', monospace;
                font-size: 16px;
                color: white;
            }
            .think {
                font-family: 'Helvetica Neue', monospace;
                font-size: 16px;
                color: orange;
            }
            </style>
            """,
            unsafe_allow_html=True)

models = [model['model'] for model in ollama.list()['models']]
embedding_models = ['nomic-embed-text']
reasoning_models = ['deepseek-r1']
vision_models = ['llama3.2-vision', 'llava']
base_llm = 'llama3.1:8b'

with st.sidebar.popover('Manage Models'):
    model = st.selectbox('Model', options=get_models())
    tag = st.selectbox('Tag', options=get_tags(model))
    c1, c2 = st.columns([1, 3])
    if c1.button('Pull'):
        if tag in models:
            with st.warning('This model already exists!') as _:
                time.sleep(1)
                st.empty()
        else:
            with st.spinner('Pulling model...', show_time=True):
                result =  ollama.pull(tag)
            if result['status'] == 'success': 
                st.success('Done!')
                st.rerun()
            else:
                st.markdown('Error!')
    if c2.button('Remove'):
        if tag not in models:
            with st.warning('This model is not downloaded!') as _:
                time.sleep(1)
                st.empty()
        else:
            with st.spinner('Pulling model...', show_time=True):
                result =  ollama.delete(tag)
            if result['status'] == 'success': 
                st.success('Done!')
                st.rerun()
            else:
                st.markdown('Error!')


model = st.sidebar.selectbox('Choose a model', models, index=0)

if 'active_context' not in ss:
    ss['active_context'] = 'New Chat'

if 'system_prompt' not in ss:
    ss.system_prompt = system_prompt_ea

if model not in st.session_state:
    ss[model] = {ss.active_context: {'messages': [{'role': 'system', 'content': ss.system_prompt}]}}

def refresh(context_name='New Chat'):
    ss[model][context_name] = {'messages': [ss[model][context_name]['messages'][0]]}

def name_context(messages):
    messages = messages + [{'role': 'user', 
                            'content': 'Generate a title for this conversation under 3 words. Do not return anything else.'}]
    return ollama.chat(model=base_llm, messages=messages)['message']['content']

def context_switch(context_name:str):

    if context_name not in ss[model]:
        new_context()
    else:
        st.session_state.active_context = context_name

def new_context():
    # TODO: Save Old Context
    if not private:
        messages = [message for message in ss[model][ss.active_context]['messages'] if message['role'] in ['user', 'assistant']]
        image = ss[model][ss.active_context].get('image', None)
        if len(messages) > 0:
            context_name = name_context(messages)
            ss[model][context_name]= {'messages': messages}
            if image:
                ss[model][context_name]['image'] = image

    # Create New Context
    refresh('New Chat')
    context_switch('New Chat')

c1, c2, c3 = st.columns([14,2,1])

c2.text('')
c2.text('')
c2.text('')
c2.text('')
private = c2.toggle('Private')

c3.text('')
c3.text('')
c3.text('')
c3.text('')
c3.button('♻️', type='tertiary', on_click=new_context)

# Styling
if model.split(':')[0] in embedding_models:
    model_type = ModelType.EMBED
    file_types = None
    c1.markdown(
                "<span class='my-title-font'>Pixie</span> <span class='embed'>Embed</span> <br> <span class='author'>by Suprateem Banerjee</span>",
                unsafe_allow_html=True
            )
elif model.split(':')[0] in reasoning_models:
    model_type = ModelType.REASON
    file_types = ['csv']
    c1.markdown(
                "<span class='my-title-font'>Pixie</span> <span class='reason'>Reason</span> <br> <span class='author'>by Suprateem Banerjee</span>",
                unsafe_allow_html=True
            )
elif model.split(':')[0] in vision_models:
    model_type = ModelType.VISION
    file_types = ['jpg', 'jpeg', 'png']
    c1.markdown(
                "<span class='my-title-font'>Pixie</span> <span class='vision'>Vision</span> <br> <span class='author'>by Suprateem Banerjee</span>",
                unsafe_allow_html=True
            )
else:
    model_type = ModelType.CHAT
    file_types = ['csv']
    c1.markdown(
                "<span class='my-title-font'>Pixie</span> <span class='chat'>Chat</span> <br> <span class='author'>by Suprateem Banerjee</span>",
                unsafe_allow_html=True
            )


st.sidebar.text_area(label='System Prompt', 
                     height=200, 
                     key='system_prompt_area', 
                     value=st.session_state.system_prompt)

sc1, sc2 = st.sidebar.columns([1,1])

if sc1.button('Executive Assistant'):
    ss.system_prompt = system_prompt_ea
    
    ss[model][ss.active_context]['messages'][0] = {'role': 'system', 'content': ss.system_prompt}
    st.rerun()

if sc2.button('Coding Assistant'):
    ss.system_prompt = system_prompt_ca
    ss[model][ss.active_context]['messages'][0] = {'role': 'system', 'content': ss.system_prompt}
    st.rerun()

if 'upload_history' not in ss:
    ss['upload_history'] = []

if file_types:
    uploaded_file = st.sidebar.file_uploader('', type=file_types)

    if uploaded_file:
        if len(ss.upload_history) == 0:
            ss.upload_history.append(uploaded_file)
        elif len(ss.upload_history) and ss.upload_history[-1] != uploaded_file:
            ss.upload_history.append(uploaded_file)
            new_context()
        if model_type == ModelType.VISION:
            if 'image' not in ss[model][ss.active_context]:
                img = Image.open(uploaded_file)
                img = ImageOps.contain(img, (800, 800))
                buffered = BytesIO()
                img.save(buffered, format='JPEG')
                ss[model][ss.active_context]['image'] = buffered.getvalue()

chats = st.sidebar.container(border=True)

for chat_name in list(ss[model].keys()):
    chats.button(chat_name, type='tertiary', on_click=context_switch, kwargs={'context_name': chat_name})

if 'image' in ss[model][ss.active_context]:
    st.image(ss[model][ss.active_context]['image'], caption='Uploaded Image')

for message in ss[model][ss.active_context]['messages']:
    if message['role'] in ['user', 'assistant']:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

user_input = st.chat_input('Type your message...')

if user_input:

    if model_type == ModelType.EMBED:
        with st.chat_message('user'):
            st.markdown(user_input)
        with st.chat_message('assistant'):
            response_container = st.empty()
            embedding = ollama.embed(model=model,
                                     input=user_input)['embeddings']
            response_container.code(embedding,
                                    wrap_lines=True,
                                    line_numbers=True,
                                    height=500)
            
    elif model_type == ModelType.REASON:
        if ss[model][ss.active_context]['messages'][0] != ss.system_prompt_area:
            ss[model][ss.active_context]['messages'][0] = {'role': 'system', 'content': ss.system_prompt_area}
        
        ss[model][ss.active_context]['messages'].append({'role': 'user', 'content': user_input})
        with st.chat_message('user'):
            st.markdown(user_input)

        with st.chat_message('assistant'):

            response_container = st.empty()
            response_text = ''
            start_time = time.time()
            
            stream = ollama.chat(model=model, messages=ss[model][ss.active_context]['messages'], stream=True)

            for chunk in stream:

                if 'message' in chunk and chunk['message'].content:
                    new_text = chunk["message"].content
                    response_text += new_text
                    response_container.markdown(response_text, unsafe_allow_html=True)
            
            _, reasoning, response = re.split(r"(<think>.*?</think>)", response_text, flags=re.DOTALL)
            response_container.empty()
            with st.expander('Reasoning'):
                st.markdown(re.sub(r"<think>(.*?)</think>", r'<span class="think">\1</span>', reasoning, flags=re.DOTALL), unsafe_allow_html=True)
            st.markdown(response)
            tps = (chunk['eval_count']) / (chunk['eval_duration'] / 1e9)
            st.caption(f'⏱️ {chunk['total_duration'] / 1e9:.2f} seconds ({tps:.2f} tokens / sec)')
        
        ss[model][ss.active_context]['messages'].append({'role': 'assistant', 'content': response})
    
    elif model_type == ModelType.VISION:

        if ss[model][ss.active_context]['messages'][0] != ss.system_prompt_area:
            ss[model][ss.active_context]['messages'][0] = {'role': 'system', 'content': ss.system_prompt_area}
        
        ss[model][ss.active_context]['messages'].append({'role': 'user', 
                                                         'content': user_input, 
                                                         'images': [ss[model][ss.active_context]['image']] 
                                                         if 'image' in ss[model][ss.active_context] else None})
        
        with st.chat_message('user'):
            st.markdown(user_input)

        with st.chat_message('assistant'):
            response_container = st.empty()
            response_text = ''

            stream = ollama.chat(model=model,
                                 messages=ss[model][ss.active_context]['messages'],
                                 stream=True)
            
            for chunk in stream:

                if 'message' in chunk and chunk['message'].content:
                    response_text += chunk['message'].content
                    response_container.markdown(response_text)
            
            tps = (chunk['eval_count']) / (chunk['eval_duration'] / 1e9)
            st.caption(f'⏱️ {chunk['total_duration'] / 1e9:.2f} seconds ({tps:.2f} tokens / sec)')
        
        ss[model][ss.active_context]['messages'].append({'role': 'assistant', 'content': response_text})
            
    else:

        if ss[model][ss.active_context]['messages'][0] != ss.system_prompt_area:
            ss[model][ss.active_context]['messages'][0] = {'role': 'system', 'content': ss.system_prompt_area}
        
        ss[model][ss.active_context]['messages'].append({'role': 'user', 'content': user_input})
        with st.chat_message('user'):
            st.markdown(user_input)

        with st.chat_message('assistant'):

            response_container = st.empty()
            response_text = ''

            stream = ollama.chat(model=model, messages=[
                    {'role': message['role'], 'content': message['content']}
                    for message in ss[model][ss.active_context]['messages']
                ], stream=True)

            for chunk in stream:

                if 'message' in chunk and chunk['message'].content:
                    response_text += chunk['message'].content
                    response_container.markdown(response_text)
            
            tps = (chunk['eval_count']) / (chunk['eval_duration'] / 1e9)
            st.caption(f'⏱️ {chunk['total_duration'] / 1e9:.2f} seconds ({tps:.2f} tokens / sec)')


        ss[model][ss.active_context]['messages'].append({'role': 'assistant', 'content': response_text})