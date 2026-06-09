import requests

url_drive = "https://drive.google.com/uc?export=download&id=1lI0FmNXDPrezPhzeryZTCEP0rl8BDuE"

headers = {
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

try:
    print("Fetching from Google Drive download link...")
    r = requests.get(url_drive, headers=headers, timeout=15)
    print(f"Status: {r.status_code}")
    print(f"Content length: {len(r.content)}")
    print(f"Content type: {r.headers.get('Content-Type')}")
    if r.status_code == 200:
        print("SUCCESS!")
        print("Preview of content (first 200 bytes):", r.content[:200])
    else:
        print(f"HTML preview: {r.text[:300]}")
except Exception as e:
    print(f"Error: {e}")
