# process_onet.py
import os
import csv
from collections import defaultdict

# update out_dir to where download_onet extracted files
OUT_DIR = "onet_text/db_30_0_text"

# filenames may vary by release; common names:
OCC_FILE = os.path.join(OUT_DIR, "Occupation Data.txt")   # may be "Occupations.txt" or similar
SKILLS_FILE = os.path.join(OUT_DIR, "Skills.txt")
TECH_FILE = os.path.join(OUT_DIR, "Technology Skills.txt")  # or "Technology Skills.txt"

# adjust if files differ in your release; list files to inspect:
print("Files in folder:", os.listdir(OUT_DIR))

# Helper to read tab-delimited; some files have different delimiters
def read_tsv(path):
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            yield line.rstrip("\n").split("\t")

# Read occupations -> map O*NET-SOC code to title
onet_to_title = {}
# Try common column positions by inspecting header manually if needed
# For many releases, occupancy file format: Onet-SOC Code (col0), Title (col1)
with open(OCC_FILE, encoding="utf-8", errors="ignore") as f:
    for line in f:
        cols = line.strip().split("\t")
        if len(cols) < 2: 
            continue
        code = cols[0].strip()
        title = cols[1].strip()
        onet_to_title[code] = title

# Parse Skills.txt -> contains skill items linked to occupation codes
# Many releases: Skills.txt fields include O*NET-SOC Code in first column, Skill name later
role_skills = defaultdict(list)  # key = onet_code -> list of skill strings

if os.path.exists(SKILLS_FILE):
    for cols in read_tsv(SKILLS_FILE):
        # Attempt to find occupation code and skill text in row
        # Different releases vary; we'll try heuristics
        if len(cols) >= 3:
            onet = cols[0].strip()
            skill = cols[2].strip()  # usually example/skill text
            if onet and skill:
                role_skills[onet].append(skill)

# Parse Technology Skills file (software, tools)
if os.path.exists(TECH_FILE):
    for cols in read_tsv(TECH_FILE):
        if len(cols) >= 2:
            onet = cols[0].strip()
            tech = cols[1].strip()
            if onet and tech:
                role_skills[onet].append(tech)

# Write a cleaned CSV: one row per occupation, skills as list (JSON-style)
import json
rows = []
for onet_code, skills in role_skills.items():
    title = onet_to_title.get(onet_code, "")
    skills_unique = sorted(set([s for s in skills if s]))
    rows.append({
        "onet_code": onet_code,
        "job_title": title,
        "skills": json.dumps(skills_unique)
    })

out_csv = "roles_skills_onet.csv"
with open(out_csv, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["onet_code", "job_title", "skills"])
    writer.writeheader()
    for r in rows:
        writer.writerow(r)

print("Wrote", out_csv, "with", len(rows), "occupations.")
