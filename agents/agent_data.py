"""
Advanced Data Agent: 사용자가 제공한 새로운 구글 드라이브 링크 반영.
"""
import os
import requests
import zipfile
from tqdm import tqdm

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)
    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        try:
            total_size = int(response.headers.get('content-length', 0))
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=os.path.basename(destination)) as pbar:
                for chunk in response.iter_content(CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        except:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk: f.write(chunk)

def extract_zip(zip_path, extract_to):
    if not os.path.exists(zip_path): return False
    print(f"[*] Extracting {zip_path} to {extract_to}...")
    try:
        os.makedirs(extract_to, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print("[+] Extraction complete.")
        return True
    except Exception as e:
        print(f"[!] Extraction failed: {e}")
        return False

def run_data_pipeline():
    print("[Data Agent] Checking external dataset paths...")
    
    datasets = {
        'PURE': 'D:\\PURE',
        'UBFC-rPPG': 'D:\\UBFC-rPPG'
    }
    
    for name, path in datasets.items():
        if os.path.exists(path):
            print(f"[Data Agent] {name} folder exists at {path}.")
        else:
            print(f"[!] [Data Agent] WARNING: {name} not found at {path}. Please check external SSD.")

if __name__ == "__main__":
    run_data_pipeline()
