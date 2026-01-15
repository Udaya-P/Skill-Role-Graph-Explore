# download_onet.py
import os
import requests
import zipfile

# Official O*NET database page: https://www.onetcenter.org/database.html
# You may need to pick specific release file (db_30_x_text.zip). Below is a generic example URL — update if needed.

ONET_ZIP_URL = "https://www.onetcenter.org/dl_files/database/db_30_0_text.zip"
OUT_ZIP = "db_onet_text.zip"
OUT_DIR = "onet_text"

os.makedirs(OUT_DIR, exist_ok=True)

print("Downloading O*NET database (this can be large)...")
r = requests.get(ONET_ZIP_URL, stream=True)
r.raise_for_status()
with open(OUT_ZIP, "wb") as f:
    for chunk in r.iter_content(1024*1024):
        f.write(chunk)

print("Extracting...")
with zipfile.ZipFile(OUT_ZIP, "r") as z:
    z.extractall(OUT_DIR)

print("Files extracted to:", OUT_DIR)
