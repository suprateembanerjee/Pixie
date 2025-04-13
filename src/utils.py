from bs4 import BeautifulSoup
import requests
import json
import requests
from bs4 import BeautifulSoup

def get_models():
    soup = BeautifulSoup(requests.get('https://ollama.com/library?sort=popular').text, features='html.parser')

    popular_models = [h2.text.strip() for h2 in soup.find_all("h2")]
    popular_models = [m for m in popular_models if m]

    return popular_models

def get_tags(model):

    model_tags = []

    response = requests.get(f'https://ollama.com/library/{model}/tags')
    response.raise_for_status()
    soup = BeautifulSoup(response.text, features='html.parser')
    for a in soup.find_all('a'):
        if not a['href'].startswith(f'/library/{model}:'):
            continue
        model_tags.append(f'{model}:{a.text.strip()}')

    return model_tags

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