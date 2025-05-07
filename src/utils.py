from bs4 import BeautifulSoup
import requests
import json
import requests
from bs4 import BeautifulSoup
import sqlite3 as lite
import ollama

from enums import ModelType
from exceptions import InsufficientArgumentsException
from models import Type

def get_models():
    soup = BeautifulSoup(requests.get('https://ollama.com/library?sort=popular').text, features='html.parser')

    popular_models = [h2.text.strip() for h2 in soup.find_all("h2")]
    popular_models = [m for m in popular_models if m]

    return popular_models

def get_tags(model):
    model_tags = []
    url = f'https://ollama.com/library/{model}/tags'
    
    response = requests.get(url)
    response.raise_for_status()
    
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find all relevant anchor tags that match the pattern
    for a in soup.select('a[href^="/library/{}:"]'.format(model)):
        href = a['href']
        tag = href.split(':')[-1].strip()  # get the part after the colon
        if tag:
            model_tags.append(f'{model}:{tag}')

    return list(dict.fromkeys(model_tags))

def scrape_ollama_model(model):
    # Send a GET request to the page
    url = f'https://ollama.com/library/{model}'
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        print(f"Failed to retrieve page, status code: {response.status_code}")
        return
    
    # Parse the page content
    soup = BeautifulSoup(response.text, features='html.parser')
    
    # Extract relevant data (Modify selectors based on actual page structure)
    title = soup.find('h1').get_text(strip=True) if soup.find('h1') else 'No title found'
    description = soup.find('meta', {'name': 'description'})
    description = description['content'] if description else 'No description found'
    
    # Extract other details if needed
    content_text = '\n'.join([p.get_text(strip=True) for p in soup.find_all('p')])
    
    # Format the output
    result = f"Title: {title}\nDescription: {description}\nContent: {content_text}"
    return result

def update_db(db_name:str,
              table:str,
              messages:dict|None=None,
              model_repository:dict|None=None):
    if table == 'models':
        if  not model_repository:
            raise InsufficientArgumentsException('model_repository')
        
        conn = lite.connect(f'file:{db_name}?mode=rw', uri=True)
        cursor = conn.cursor()
        cursor.execute('''DROP TABLE IF EXISTS models''')
        cursor.execute('''CREATE TABLE models (model TEXT, type TEXT)''')
        for model_type, model_families in model_repository.items():
            for model_family in model_families:
                cursor.execute(f"INSERT INTO models (model, type) VALUES ('{model_family}', '{model_type.value}')")
        conn.commit()
        conn.close()

    elif table == 'context':
        if not messages:
            raise InsufficientArgumentsException('messages')
        conn = lite.connect(f'file:{db_name}?mode=rw', uri=True)
        cursor = conn.cursor()
        cursor.execute('''DROP TABLE IF EXISTS context''')
        cursor.execute('''CREATE TABLE context (model TEXT, context TEXT, message TEXT)''')
        for model in messages:
            for context, messages_ in messages[model].items():
                cursor.execute(f'INSERT INTO context (model, context, message) VALUES (?, ?, ?)', (model, context, json.dumps(messages_)))
        conn.commit()
        conn.close()

def load_from_db(table:str,
                 db_name:str) -> dict:
    if table == 'models':
        # Intialize Repository
        model_repository = {}
        model_repository[ModelType.REASON] = []
        model_repository[ModelType.CHAT] = []
        model_repository[ModelType.EMBED] = []
        model_repository[ModelType.VISION] = []

        conn = lite.connect(f'file:{db_name}?mode=rw', uri=True)
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM models')
        models_ = cursor.fetchall()

        for model_, type_ in models_:
            model_repository[ModelType(type_)].append(model_)

        return model_repository

    elif table == 'context':
        conn = lite.connect(f'file:{db_name}?mode=rw', uri=True)
        cursor = conn.cursor()
        messages = {}
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='context';")
        if cursor.fetchone():
            cursor.execute(f'SELECT * FROM {table}')
            contexts = cursor.fetchall()
            for model_, context_, messages_ in contexts:
                if model_ not in messages:
                    messages[model_] = {}
                if context_ not in messages[model_]:
                    messages[model_][context_] = json.loads(messages_)
        conn.commit()
        conn.close()

        return messages
    
def create_repository(table:str,
                      db_name:str,
                      llm:str,
                      models:list[str]):

    conn = lite.connect(db_name)
    cursor = conn.cursor()
    cursor.execute(f'''CREATE TABLE IF NOT EXISTS {table} (model TEXT, type TEXT)''')
    model_repository = {}
    model_repository[ModelType.REASON] = []
    model_repository[ModelType.CHAT] = []
    model_repository[ModelType.EMBED] = []
    model_repository[ModelType.VISION] = []

    for model in models:
        summary = scrape_ollama_model(model)
        model_type = ollama.chat(model=llm, 
                                messages=[{'role': 'system', 
                                            'content': 'Pick the correct type of model based on description'},
                                            {'role':'user',
                                            'content':summary}],
                                format=Type.model_json_schema())['message']['content']
        model_type = ModelType(json.loads(model_type)['model_type'])
        model_repository[ModelType(model_type)].append(model.split(':')[0])
        cursor.execute(f"INSERT INTO models (model, type) VALUES ('{model.split(':')[0]}', '{model_type.value}')")
    conn.commit()
    conn.close()

    return model_repository