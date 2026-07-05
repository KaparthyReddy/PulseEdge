import os
import requests

BASE_URL = "https://engineering.case.edu/sites/default/files/"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

FILES = {
    "normal_baseline.mat": "97.mat",
    "inner_race_007.mat": "105.mat",
    "ball_007.mat": "118.mat",
    "outer_race_007.mat": "130.mat",
}

def download():
    os.makedirs(DATA_DIR, exist_ok=True)
    for local_name, remote_name in FILES.items():
        path = os.path.join(DATA_DIR, local_name)
        if os.path.exists(path):
            print(f"Skipping {local_name}, already exists")
            continue
        url = BASE_URL + remote_name
        print(f"Downloading {url}")
        r = requests.get(url, stream=True)
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Saved to {path}")

if __name__ == "__main__":
    download()
