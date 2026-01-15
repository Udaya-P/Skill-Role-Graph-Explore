# Skill–Role Graph Explorer

A **graph-based career recommendation system** that models relationships between **job roles and skills** using **graph embeddings**. The system enables intelligent **role-to-skill**, **skill-to-role**, and **career transition** recommendations with interactive visualizations.

---

## 🚀 Project Overview

Traditional career recommendation systems rely on keyword matching or static skill lists.  
**Skill–Role Graph Explorer** uses **graph representation learning** to capture how skills co-occur, cluster around roles, and enable realistic career transitions.

The system:
- Represents **roles and skills as nodes** in a heterogeneous graph
- Learns **structure-aware embeddings** using Node2Vec
- Predicts **relevant skills for a role** and **best-fit roles for a skill set**
- Identifies **missing skills** required to transition between roles
- Visualizes skill and role relationships interactively

---

## ✨ Key Features

- Role → Skill recommendation
- Skill → Role prediction
- Career transition modeling
- Missing skill identification
- Graph-based similarity analysis
- Interactive visualizations of skill & role graphs

---

## 🧠 Technologies Used

- **Programming Language:** Python  
- **Framework:** Flask  
- **Graph Modeling:** NetworkX  
- **Embeddings:** Node2Vec, Word2Vec  
- **Machine Learning:** Scikit-learn  
- **Visualization:** D3.js  
- **Version Control:** Git & GitHub  

---

## 🏗️ System Architecture

1. Build a heterogeneous graph of roles and skills
2. Generate embeddings using Node2Vec
3. Compute similarity using cosine distance
4. Predict skills, roles, and transitions
5. Visualize graphs through a web interface

---

## 📂 Project Structure

-Role-Graph-Explore
Skill-Role-Graph-Explore/
│── app.py
│── graph_builder.py
│── embeddings.py
│── recommender.py
│── static/
│── templates/
│── data/
│── README.md


---

## How to Run the Project

```bash

1) Run the Application
python app.py

2) Open in Browser
http://localhost:5000
