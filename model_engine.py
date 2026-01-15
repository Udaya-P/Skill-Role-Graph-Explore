import pandas as pd
import networkx as nx
from gensim.models import KeyedVectors, Word2Vec
import numpy as np

CSV_PATH = "roles_skills_generated.csv"
GML_PATH = "skill_role_graph.gml"
N2V_PATH = "node2vec_embeddings.kv"
W2V_PATH = "word2vec.model"


# ----------------------------
# LOAD GRAPH + MODELS
# ----------------------------
def load_all():
    df = pd.read_csv(CSV_PATH)
    G = nx.read_gml(GML_PATH)

    n2v = KeyedVectors.load(N2V_PATH)
    w2v = Word2Vec.load(W2V_PATH)

    return df, G, n2v, w2v


# ----------------------------
# PREDICT SKILLS FROM ROLE
# ----------------------------
def predict_skills(role, n2v, G, top_k=10):
    if role not in n2v:
        return []

    similar = n2v.most_similar(role, topn=top_k * 3)
    skills = [n for n, score in similar if G.nodes[n]["type"] == "skill"]
    return skills[:top_k]


# ----------------------------
# PREDICT ROLE FROM SKILLS
# ----------------------------
def predict_role(skill_list, n2v, G, top_k=5):
    scores = {}
    roles = [n for n, d in G.nodes(data=True) if d["type"] == "role"]

    for s in skill_list:
        if s not in n2v:
            continue

        for r in roles:
            if r in n2v:
                sim = n2v.similarity(s, r)
                scores[r] = scores.get(r, 0) + sim

    sorted_roles = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_roles[:top_k]

def get_role_list(G):
    return [n for n, d in G.nodes(data=True) if d.get("type") == "role"]

def build_role_embedding_matrix(n2v, G):
    roles = get_role_list(G)
    vecs = []
    filtered_roles = []

    for r in roles:
        if r in n2v:
            vecs.append(n2v[r])
            filtered_roles.append(r)

    if len(vecs) == 0:
        return filtered_roles, np.zeros((0, n2v.vector_size))

    return filtered_roles, np.vstack(vecs)

def role_neighbors(role, n2v, G, top_k=5):
    if role not in n2v:
        return []

    raw = n2v.most_similar(role, topn=top_k * 5)

    out = []
    for node, score in raw:
        if node == role:
            continue
        if G.nodes[node]["type"] == "role":
            out.append((node, float(score)))
        if len(out) >= top_k:
            break

    return out

def auto_generate_transitions(n2v, G, top_k=3, similarity_threshold=None):
    """
    Returns list of transitions:
    [
        (source_role, target_role, similarity_score)
    ]
    """

    roles = get_role_list(G)
    transitions = []

    for r in roles:
        neighs = role_neighbors(r, n2v, G, top_k=top_k)
        for tgt, score in neighs:
            if similarity_threshold and score < similarity_threshold:
                continue
            transitions.append((r, tgt, score))

    return transitions

def get_skills_for_role(df, role_name):
    return set(df[df["role"] == role_name]["skill"].dropna().unique().tolist())

def recommend_missing_skills(df, source_role, target_role, top_n=10):
    src_skills = get_skills_for_role(df, source_role)
    tgt_skills = get_skills_for_role(df, target_role)

    missing = list(tgt_skills - src_skills)
    if not missing:
        return []

    # Rank missing skills by frequency within target role
    freq = df[df["role"] == target_role]["skill"].value_counts().to_dict()

    missing_sorted = sorted(missing, key=lambda s: freq.get(s, 0), reverse=True)
    return missing_sorted[:top_n]