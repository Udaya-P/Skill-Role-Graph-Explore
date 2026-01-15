# role_skill_engine_onet.py
import pandas as pd
import json
from rapidfuzz import process, fuzz
from collections import Counter
import math

CSV = "roles_skills_onet.csv"

def load_roles(csv_path=CSV):
    df = pd.read_csv(csv_path)
    # ensure skills column is python list
    df["skills"] = df["skills"].apply(lambda x: json.loads(x) if isinstance(x, str) else [])
    return df

def match_role(df, query, cutoff=60):
    roles = df["job_title"].fillna("").tolist()
    match, score, _ = process.extractOne(query, roles, scorer=fuzz.token_set_ratio)
    if score < cutoff:
        return None, score
    # find first matching row
    row = df[df["job_title"] == match].iloc[0]
    return row, score

# Distinctive scoring: P(skill|role) - P(skill|others)
def get_distinctive_skills_for_role(df, role_row, top_n=30):
    role_skills = Counter([s.lower() for s in role_row["skills"]])
    # others
    other_rows = df[df["onet_code"] != role_row["onet_code"]]
    other_skills = Counter()
    for s_list in other_rows["skills"]:
        other_skills.update([s.lower() for s in s_list])

    total_role = sum(role_skills.values()) or 1
    total_other = sum(other_skills.values()) or 1

    scores = {}
    for s, cnt in role_skills.items():
        p_role = cnt / total_role
        p_other = (other_skills.get(s, 0) / total_other)
        scores[s] = p_role - p_other

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_n]

# Frequency listing (simple)
def get_frequency_skills_for_role(role_row, top_n=30):
    freq = Counter([s.lower() for s in role_row["skills"]])
    return freq.most_common(top_n)

if __name__ == "__main__":
    df = load_roles()
    r, sc = match_role(df, "machine learning engineer")
    print("Matched:", r["job_title"], sc)
    print("Top distinctive:", get_distinctive_skills_for_role(df, r)[:20])
    print("Top freq:", get_frequency_skills_for_role(r)[:20])
