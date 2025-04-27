import streamlit as st
from streamlit import session_state as ss
import ollama
import time
import re
from PIL import Image, ImageOps
from io import BytesIO
import pandas as pd
import json
import sqlite3 as lite
import base64
from urllib.request import pathname2url

from prompts import system_prompt_ea, system_prompt_ca
from utils import get_models, get_tags, scrape_ollama_model, load_from_db, create_repository, update_db
from enums import ModelType
from models import Type
from exceptions import InsufficientArgumentsException

st.set_page_config(page_title='Pixie', layout='wide')
Image.MAX_IMAGE_PIXELS = None

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

models = sorted([model['model'] for model in ollama.list()['models']])

if 'base_llm' not in ss:
    ss.base_llm = 'llama3.1:8b'

ss.db = {'persistent_repository': 'persistent_repository.db'}

if 'model_repository' not in ss:
    try:
        ss['model_repository'] = load_from_db(table='models', 
                                              db_name=ss.db['persistent_repository'])
    except lite.OperationalError:
        with st.spinner('Creating Model Repository...', show_time=True):
            ss['model_repository'] = create_repository(table='models', 
                                                       db_name=ss.db['persistent_repository'], 
                                                       llm=ss.base_llm, 
                                                       models=models)

if 'messages' not in ss:
    ss['messages'] = load_from_db(table='context', db_name=ss.db['persistent_repository'])

with st.sidebar.popover('Manage Models'):
    model_family = st.selectbox('Model', options=get_models())
    tag = st.selectbox('Tag', options=get_tags(model_family))
    c1, c2 = st.columns([1, 3])
    if c1.button('Pull', disabled=tag in models):
        with st.spinner('Pulling model...', show_time=True):
            result =  ollama.pull(tag)
        if result['status'] == 'success':
            add_to_repository = True

            for model_type_, model_families in ss.model_repository.items():
                if model_family in model_families:
                    add_to_repository = False
                    break
            
            if add_to_repository:
                summary = scrape_ollama_model(tag)
                model_type = ollama.chat(model=ss.base_llm, 
                                        messages=[{'role': 'system', 
                                                    'content': 'Pick the correct type of model based on description'},
                                                    {'role':'user',
                                                    'content':summary}],
                                        format=Type.model_json_schema())['message']['content']
                
                model_type = ModelType(json.loads(model_type)['model_type'])
                if model_family not in ss.model_repository[model_type]:
                    ss.model_repository[model_type].append(model_family)

            update_db(table='models', model_repository=ss.model_repository, db_name=ss.db['persistent_repository'])
            st.success('Done!')
            st.rerun()
        else:
            st.warning('Error!')
    if c2.button('Remove', disabled=tag not in models):
        with st.spinner('Removing model...', show_time=True):
            result =  ollama.delete(tag)
        if result['status'] == 'success': 

            remove_from_repository = True
            model_family = tag.split(':')[0]

            for downloaded_model in ollama.list()['models']:
                if model_family == downloaded_model['model'].split(':')[0]:
                    remove_from_repository = False
                    break

            if remove_from_repository:
                model_type = None

                for model_type_, model_families in ss.model_repository.items():
                    if model_family in model_families:
                        model_type = model_type_
                        break
                ss.model_repository[model_type].remove(model_family)

            update_db(table='models', model_repository=ss.model_repository, db_name=ss.db['persistent_repository'])
            st.success('Done!')
            st.rerun()
        else:
            st.markdown('Error!')

    st.divider()

    chat_models = []
    for model_family in ss.model_repository[ModelType.CHAT]:
        for candidate in models:
            if candidate[:len(model_family)] == model_family:
                chat_models.append(candidate)

    st.selectbox(label='Base LLM', options=chat_models, index=chat_models.index(ss.base_llm), key='base_llm')

    rows = []
    for model_type, available_models in ss.model_repository.items():
        for model in list(set(available_models)):
            rows.append({'Model': model, 'Type': model_type.value})
    
    df = pd.DataFrame(rows)
    df['Type'] = pd.Categorical(df['Type'], categories=[mt.value for mt in ModelType])
    df.sort_values(by='Model', inplace=True)

    edited_df = st.data_editor(df, hide_index=True, use_container_width=True)

    model_repository = {}
    model_repository[ModelType.REASON] = []
    model_repository[ModelType.CHAT] = []
    model_repository[ModelType.EMBED] = []
    model_repository[ModelType.VISION] = []

    for model_type in list(set(list(edited_df['Type']))):
        model_repository[ModelType(model_type)] = list(edited_df[edited_df['Type'] == model_type]['Model'])
    
    if ss.model_repository != model_repository:
        ss.model_repository = model_repository
        update_db(table='models', model_repository=ss.model_repository, db_name=ss.db['persistent_repository'])

st.sidebar.selectbox('Choose a model', models, index=0, key='model')

if 'system_prompt' not in ss:
    ss.system_prompt = system_prompt_ea

if 'active_context' not in ss:
    ss['active_context'] = 'New Chat'

if ss.model not in ss.messages:
    ss.active_context = 'New Chat'
    ss.messages[ss.model] = {ss.active_context: {'messages': [{'role': 'system', 'content': ss.system_prompt}]}}

if ss.active_context not in ss.messages[ss.model]:
    ss.active_context = 'New Chat'

def refresh(context_name='New Chat'):
    ss.messages[ss.model][context_name] = {'messages': [{'role': 'system', 'content': ss.system_prompt}]}

def name_context(messages):
    messages = messages + [{'role': 'user', 
                            'content': 'Generate a title for this conversation under 3 words. Do not return anything else.'}]
    return ollama.chat(model=ss.base_llm, messages=messages)['message']['content']

def context_switch(context_name:str):

    if context_name not in ss.messages[ss.model]:
        new_context()
    else:
        st.session_state.active_context = context_name
        update_db(table='context', messages=ss.messages, db_name=ss.db['persistent_repository'])

def new_context():
    # If active context is not new context, switch to new context
    if ss.active_context != 'New Chat':
        ss.active_context = 'New Chat'
    # Save Old Context
    if len(ss.messages[ss.model][ss.active_context]['messages']) > 1:
        messages = [message for message in ss.messages[ss.model][ss.active_context]['messages'] if message['role'] in ['user', 'assistant']]
        image = ss.messages[ss.model][ss.active_context].get('image', None)
        if len(messages) > 0:
            context_name = name_context(messages)
            ss.messages[ss.model][context_name]= {'messages': messages}
            if image:
                ss.messages[ss.model][context_name]['image'] = image

    # Create New Context
    refresh('New Chat')
    context_switch('New Chat')

c1, c2, c3 = st.columns([14,2,1])

c3.text('')
c3.text('')
c3.text('')
c3.text('')
c3.button('♻️', type='tertiary', on_click=new_context)

# Styling
if ss.model.split(':')[0] in ss.model_repository[ModelType.EMBED]:
    model_type = ModelType.EMBED
    file_types = None
    c1.markdown(
                "<span class='my-title-font'>Pixie</span> <span class='embed'>Embedding</span> <br> <span class='author'>by Suprateem Banerjee</span>",
                unsafe_allow_html=True
            )
elif ss.model.split(':')[0] in ss.model_repository[ModelType.REASON]:
    model_type = ModelType.REASON
    file_types = ['csv']
    c1.markdown(
                "<span class='my-title-font'>Pixie</span> <span class='reason'>Reason</span> <br> <span class='author'>by Suprateem Banerjee</span>",
                unsafe_allow_html=True
            )
elif ss.model.split(':')[0] in ss.model_repository[ModelType.VISION]:
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

if 'upload_history' not in ss:
    ss['upload_history'] = []

if file_types:
    uploaded_file = st.sidebar.file_uploader('Upload Files', type=file_types, label_visibility='collapsed')

    if uploaded_file:
        if len(ss.upload_history) == 0:
            ss.upload_history.append(uploaded_file)
        elif len(ss.upload_history) and ss.upload_history[-1] != uploaded_file:
            ss.upload_history.append(uploaded_file)
            new_context()
        if model_type == ModelType.VISION:
            if 'image' not in ss.messages[ss.model][ss.active_context]:
                img = Image.open(uploaded_file)
                img = ImageOps.contain(img, (800, 800))
                buffered = BytesIO()
                img.save(buffered, format='JPEG')
                ss.messages[ss.model][ss.active_context]['image'] = base64.b64encode(buffered.getvalue()).decode('utf-8')

chats = st.sidebar.container(border=True)
c1, c2, c3 = chats.columns([8,1,1])

if 'context_to_edit' not in ss:
    ss.context_to_edit = None

if 'context_edit_values' not in ss:
    ss.context_edit_values = {}

def delete_context(context:str):
    if len(ss.messages[ss.model]) > 1:
        ss.messages[ss.model].pop(context)
        update_db(table='context', messages=ss.messages, db_name=ss.db['persistent_repository'])
        ss.active_context = 'New Chat'
    else:
        refresh()

def edit_context(context:str):
    ss.context_to_edit = context

def apply_edit_context():
    old_context = ss.context_to_edit
    new_context_name = ss[f'rename_input_{old_context}']
    if old_context and new_context_name:
        ss.messages[ss.model][new_context_name] = ss.messages[ss.model].pop(old_context)
        ss.active_context = new_context_name
        update_db(table='context', messages=ss.messages, db_name=ss.db['persistent_repository'])
    ss.context_to_edit = None

def reset_edit_context():
    ss.context_to_edit = None

for context in ss.messages[ss.model]:
    if context == 'New Chat':
        c1.button(context, type='tertiary', on_click=context_switch, kwargs={'context_name': context})
        c2.button('', type='tertiary', key=f'edit_{context}')
        c3.button('', type='tertiary', key=f'close_{context}')
    elif ss.context_to_edit == context:
        c1.text_input('', value=context, key=f'rename_input_{context}', label_visibility='collapsed', on_change=apply_edit_context)
        c2.button('', type='tertiary', key=f'edit_{context}')
        c3.button('', type='tertiary', key=f'close_{context}')
    else:
        c1.button(context, type='tertiary', on_click=context_switch, kwargs={'context_name': context})
        c2.button('✎', type='tertiary', key=f'edit_{context}', on_click=edit_context, kwargs={'context': context})
        c3.button('×', type='tertiary', key=f'close_{context}', on_click=delete_context, kwargs={'context': context})

if 'image' in ss.messages[ss.model][ss.active_context]:
    st.image(base64.b64decode(ss.messages[ss.model][ss.active_context]['image']), caption='Uploaded Image')

for message in ss.messages[ss.model][ss.active_context]['messages']:
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
            embedding = ollama.embed(model=ss.model,
                                     input=user_input)['embeddings']
            response_container.code(embedding,
                                    wrap_lines=True,
                                    line_numbers=True,
                                    height=500)
            
    elif model_type == ModelType.REASON:
        if ss.messages[ss.model][ss.active_context]['messages'][0] != ss.system_prompt_area:
            ss.messages[ss.model][ss.active_context]['messages'][0] = {'role': 'system', 'content': ss.system_prompt_area}
        
        ss.messages[ss.model][ss.active_context]['messages'].append({'role': 'user', 'content': user_input})
        with st.chat_message('user'):
            st.markdown(user_input)

        with st.chat_message('assistant'):

            response_container = st.empty()
            response_text = ''
            start_time = time.time()
            
            stream = ollama.chat(model=ss.model, messages=ss.messages[ss.model][ss.active_context]['messages'], stream=True)

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
        
        ss.messages[ss.model][ss.active_context]['messages'].append({'role': 'assistant', 'content': response})
    
    elif model_type == ModelType.VISION:

        if ss.messages[ss.model][ss.active_context]['messages'][0] != ss.system_prompt_area:
            ss.messages[ss.model][ss.active_context]['messages'][0] = {'role': 'system', 'content': ss.system_prompt_area}
        
        ss.messages[ss.model][ss.active_context]['messages'].append({'role': 'user', 
                                                         'content': user_input, 
                                                         'images': [ss.messages[ss.model][ss.active_context]['image']] 
                                                         if 'image' in ss.messages[ss.model][ss.active_context] else None})
        
        with st.chat_message('user'):
            st.markdown(user_input)

        with st.chat_message('assistant'):
            response_container = st.empty()
            response_text = ''

            stream = ollama.chat(model=ss.model,
                                 messages=ss.messages[ss.model][ss.active_context]['messages'],
                                 stream=True)
            
            for chunk in stream:

                if 'message' in chunk and chunk['message'].content:
                    response_text += chunk['message'].content
                    response_container.markdown(response_text)
            
            tps = (chunk['eval_count']) / (chunk['eval_duration'] / 1e9)
            st.caption(f'⏱️ {chunk['total_duration'] / 1e9:.2f} seconds ({tps:.2f} tokens / sec)')
        
        ss.messages[ss.model][ss.active_context]['messages'].append({'role': 'assistant', 'content': response_text})
            
    else:

        if ss.messages[ss.model][ss.active_context]['messages'][0] != ss.system_prompt_area:
            ss.messages[ss.model][ss.active_context]['messages'][0] = {'role': 'system', 'content': ss.system_prompt_area}
        
        ss.messages[ss.model][ss.active_context]['messages'].append({'role': 'user', 'content': user_input})
        with st.chat_message('user'):
            st.markdown(user_input)

        with st.chat_message('assistant'):

            response_container = st.empty()
            response_text = ''

            stream = ollama.chat(model=ss.model, messages=[
                    {'role': message['role'], 'content': message['content']}
                    for message in ss.messages[ss.model][ss.active_context]['messages']
                ], stream=True)

            for chunk in stream:

                if 'message' in chunk and chunk['message'].content:
                    response_text += chunk['message'].content
                    response_container.markdown(response_text)
            
            tps = (chunk['eval_count']) / (chunk['eval_duration'] / 1e9)
            st.caption(f'⏱️ {chunk['total_duration'] / 1e9:.2f} seconds ({tps:.2f} tokens / sec)')


        ss.messages[ss.model][ss.active_context]['messages'].append({'role': 'assistant', 'content': response_text})