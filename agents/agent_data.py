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
    os.makedirs('data', exist_ok=True)
    # 사용자가 제공한 새로운 ID 반영
    datasets = {
        'PURE': {'id': '1dxD19tZWcTb2l75lt8dAU_xbbtqPraYh', 'path': 'data/PURE.zip', 'dir': 'data/PURE'},
        'UBFC-rPPG': {'id': '1CRJRtx1wFQ_ZDChmorZefHv0Vf_vNmcG', 'path': 'data/UBFC.zip', 'dir': 'data/UBFC'}
    }
    for name, info in datasets.items():
        is_empty = not os.path.exists(info['dir']) or not os.listdir(info['dir'])
        if is_empty:
            print(f"\n[Data Agent] Downloading {name} (New Link)...")
            download_file_from_google_drive(info['id'], info['path'])
            extract_zip(info['path'], info['dir'])
            if os.path.exists(info['path']): os.remove(info['path'])
        else:
            print(f"[Data Agent] {name} folder exists.")

if __name__ == "__main__":
    run_data_pipeline()
