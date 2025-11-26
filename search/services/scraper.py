import os
import argparse
import requests
import json
import time
import re
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, urldefrag
from rich.console import Console
from datetime import datetime

console = Console()

# Config
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, "data")
HEADERS = {'User-Agent': 'StrataSearch-Bot/1.0'}

def clean_content(soup):
    # Remove noise
    for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'iframe']):
        tag.decompose()
    
    # Extract code blocks specifically to preserve formatting
    code_blocks = []
    for pre in soup.find_all('pre'):
        code_blocks.append(pre.get_text())
        
    text = soup.get_text(separator='\n', strip=True)
    return text, len(code_blocks)

def save_page(data, folder_name):
    target_dir = os.path.join(DATA_PATH, folder_name)
    os.makedirs(target_dir, exist_ok=True)
    
    # Create a safe filename
    safe_title = re.sub(r'[^\w\s-]', '', data['title']).strip().lower()
    safe_title = re.sub(r'[-\s]+', '-', safe_title)[:50]
    file_hash = str(hash(data['url']))[-6:]
    filename_base = f"{safe_title}-{file_hash}"
    
    # 1. Save Raw Text (for vector embedding)
    txt_path = os.path.join(target_dir, f"{filename_base}.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"URL: {data['url']}\n\n{data['content']}")
        
    # 2. Save Rich Metadata (JSON) - NEW FEATURE
    json_path = os.path.join(target_dir, f"{filename_base}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
        
    return txt_path

def scrape_url(start_url, max_pages=10):
    visited = set()
    queue = [start_url]
    domain = urlparse(start_url).netloc
    folder_name = domain.replace('.', '_')
    
    count = 0
    while queue and count < max_pages:
        url = queue.pop(0)
        if url in visited: continue
        
        try:
            resp = requests.get(url, headers=HEADERS, timeout=10)
            if resp.status_code != 200: continue
            
            soup = BeautifulSoup(resp.content, 'html.parser')
            content, code_count = clean_content(soup)
            
            if len(content.split()) < 50: continue # Skip empty pages
            
            page_data = {
                'url': url,
                'title': soup.title.string if soup.title else 'Untitled',
                'content': content,
                'code_blocks_count': code_count,
                'scraped_at': datetime.now().isoformat()
            }
            
            save_path = save_page(page_data, folder_name)
            console.print(f"[green]✔ Saved:[/green] {url} [dim]({code_count} code blocks)[/dim]")
            
            visited.add(url)
            count += 1
            
            # Find links
            for a in soup.find_all('a', href=True):
                full_link = urljoin(url, a['href'])
                if urlparse(full_link).netloc == domain and full_link not in visited:
                    queue.append(full_link)
                    
            time.sleep(0.5)
            
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        scrape_url(sys.argv[1])
    else:
        print("Usage: python scraper.py <url>")