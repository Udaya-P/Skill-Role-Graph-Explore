#!/usr/bin/env python3
"""Skill-Role Graph Pipeline
Generates a skill-role graph, trains Node2Vec and Word2Vec embeddings,
provides prediction utilities and evaluates metrics.

Input CSV (role,skill): roles_skills_generated.csv

Run: python skill_role_pipeline.py
"""
import argparse
import random
from pathlib import Path
import pandas as pd
import networkx as nx
from itertools import combinations
import numpy as np

# Optional: For embeddings
try:
    from node2vec import Node2Vec
except Exception as e:
    raise SystemExit("Missing 'node2vec' package. Install via: pip install node2vec") from e

from gensim.models import Word2Vec

CSV_PATH = "roles_skills_generated.csv"
GML_OUT = "skill_role_graph.gml"
N2V_MODEL_OUT = "node2vec_embeddings.kv"
W2V_MODEL_OUT = "word2vec.model"

def build_graph(df):
    G = nx.Graph()
    for r, s in df[['role', 'skill']].values:
        G.add_node(r, type="role")
        G.add_node(s, type="skill")
        G.add_edge(r, s, weight=1)
    # skill-skill co-occurrence edges
    for role, group in df.groupby("role"):
        skills = group['skill'].tolist()
        for s1, s2 in combinations(skills, 2):
            if G.has_edge(s1, s2):
                G[s1][s2]['weight'] += 1
            else:
                G.add_edge(s1, s2, weight=1)
    return G

def train_node2vec(G, dimensions=64, walk_length=20, num_walks=100, workers=2, window=10):
    node2vec = Node2Vec(G, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, workers=workers, quiet=True)
    model = node2vec.fit(window=window, min_count=1)
    model.wv.save(N2V_MODEL_OUT)
    return model

def train_word2vec(df, vector_size=64):
    sentences = df.groupby("role")["skill"].apply(list).tolist()
    model = Word2Vec(sentences, vector_size=vector_size, window=5, min_count=1, workers=4)
    model.save(W2V_MODEL_OUT)
    return model

def predict_skills_from_role(role, model_n2v, G, top_k=10):
    if role not in model_n2v.wv:
        return []
    similar = model_n2v.wv.most_similar(role, topn=top_k*2) # fetch extra, filter skills
    skills = [n for n,score in similar if G.nodes[n].get('type')=='skill']
    return skills[:top_k]

def predict_role_from_skills(skills, model_n2v, G, top_k=5):
    scores = {}
    roles = [n for n,d in G.nodes(data=True) if d.get('type')=='role']
    for skill in skills:
        if skill not in model_n2v.wv:
            continue
        for role in roles:
            if role in model_n2v.wv:
                sim = model_n2v.wv.similarity(skill, role)
                scores[role] = scores.get(role, 0.0) + sim
    sorted_roles = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_roles[:top_k]

def evaluate_role_to_skills(df, model_n2v, G, k=5):
    role_groups = df.groupby("role")["skill"].apply(list).to_dict()
    precisions = []
    recalls = []

    for role, true_skills in role_groups.items():
        # Predict skills from role
        preds = predict_skills_from_role(role, model_n2v, G, top_k=k)

        true_set = set(true_skills)
        pred_set = set(preds)

        if len(preds) == 0:
            continue

        prec = len(pred_set & true_set) / len(pred_set)
        rec = len(pred_set & true_set) / len(true_set)

        precisions.append(prec)
        recalls.append(rec)

    return (np.mean(precisions) if precisions else 0.0,
            np.mean(recalls) if recalls else 0.0)


def evaluate_skills_to_role(df, model_n2v, G, trials_per_role=10):
    # For each role, sample a subset of its skills and try to predict the role. Measure top-1 accuracy.
    role_groups = df.groupby("role")["skill"].apply(list).to_dict()
    total = 0
    correct = 0
    for role, skills in role_groups.items():
        if len(skills) < 2:
            continue
        for _ in range(trials_per_role):
            sample_size = min(max(1, len(skills)//2), 4)
            sample = random.sample(skills, sample_size)
            preds = predict_role_from_skills(sample, model_n2v, G, top_k=3)
            total += 1
            if len(preds)>0 and preds[0][0] == role:
                correct += 1
    acc = correct/total if total>0 else 0.0
    return acc

def main():
    print("Loading CSV:", CSV_PATH)
    df = pd.read_csv(CSV_PATH)
    G = build_graph(df)
    nx.write_gml(G, GML_OUT)
    print("Graph built -- nodes:", G.number_of_nodes(), "edges:", G.number_of_edges())

    print("Training Node2Vec... (this may take a minute)")
    model_n2v = train_node2vec(G, dimensions=64, walk_length=20, num_walks=100, workers=2)
    print("Node2Vec saved to:", N2V_MODEL_OUT)

    print("Training Word2Vec on role->skill sentences...")
    model_w2v = train_word2vec(df, vector_size=64)
    print("Word2Vec saved to:", W2V_MODEL_OUT)

    # Quick examples
    example_role = df['role'].iloc[0]
    print("\nExample role:", example_role)
    print("Predicted skills (top 5):", predict_skills_from_role(example_role, model_n2v, G, top_k=5))

    print("\nEvaluating role->skills (precision@5, recall@5)...")
    prec, rec = evaluate_role_to_skills(df, model_n2v, G, k=5)
    print("Precision@5 (mean):", round(prec, 4))
    print("Recall@5 (mean):", round(rec, 4))

    print("\nEvaluating skills->role (top-1 accuracy via sampling)...")
    acc = evaluate_skills_to_role(df, model_n2v, G, trials_per_role=20)
    print("Role prediction accuracy (top-1):", round(acc, 4))

if __name__ == '__main__':
    main()
