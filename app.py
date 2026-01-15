from flask import Flask, render_template, request, jsonify
from model_engine import load_all, predict_skills, predict_role, auto_generate_transitions, get_role_list, recommend_missing_skills

app = Flask(__name__)

# Load models and graph
df, G, n2v, w2v = load_all()

roles = sorted(df["role"].unique().tolist())
skills = sorted(df["skill"].unique().tolist())


@app.route("/", methods=["GET", "POST"])
def index():
    selected_role = None
    predicted_skills = []
    predicted_roles = []

    if request.method == "POST":

        # ROLE → SKILL prediction
        if "role_select" in request.form:
            selected_role = request.form.get("role_select")
            predicted_skills = predict_skills(selected_role, n2v, G, top_k=10)

        # SKILL → ROLE prediction
        if "skills_input" in request.form:
            entered = request.form.get("skills_input")
            skill_list = [x.strip() for x in entered.split(",") if x.strip()]
            predicted_roles = predict_role(skill_list, n2v, G, top_k=5)

    return render_template(
        "index.html",
        roles=roles,
        skills=skills,
        selected_role=selected_role,
        predicted_skills=predicted_skills,
        predicted_roles=predicted_roles
    )

@app.route("/graph")
def graph_page():
    return render_template("graph.html")


@app.route("/graph-data")
def graph_data():
    nodes = []
    edges = []

    for n, data in G.nodes(data=True):
        nodes.append({"id": n, "type": data["type"]})

    for u, v, data in G.edges(data=True):
        edges.append({"source": u, "target": v})

    return {"nodes": nodes, "edges": edges}

# -------------------------------------------
# CAREER TRANSITION GRAPH  (D3 Visualization)
# -------------------------------------------
@app.route("/career-graph")
def career_graph_page():
    return render_template("career_graph.html")


@app.route("/career-graph-data")
def career_graph_data():
    # Generate transitions using embeddings
    transitions = auto_generate_transitions(n2v, G, top_k=3)

    # Build JSON nodes (roles only)
    nodes = [{"id": r, "type": "role"} for r in get_role_list(G)]

    # Build JSON directed edges
    edges = [
        {"source": src, "target": tgt, "weight": float(score)}
        for src, tgt, score in transitions
    ]

    return jsonify({"nodes": nodes, "edges": edges})


# -------------------------------------------
# CAREER PATH RECOMMENDER
# -------------------------------------------
@app.route("/career-path", methods=["GET", "POST"])
def career_path_page():
    selected_role = None
    suggestions = []
    missing_map = {}

    if request.method == "POST":
        selected_role = request.form.get("role_select")

        # Get auto next roles for this role
        transitions = auto_generate_transitions(n2v, G, top_k=5)

        # Filter only transitions starting from selected role
        suggestions = [(tgt, score) for src, tgt, score in transitions if src == selected_role]

        # Missing skills for each suggested transition
        for tgt, score in suggestions:
            missing = recommend_missing_skills(df, selected_role, tgt, top_n=10)
            missing_map[tgt] = missing

    roles_sorted = sorted(get_role_list(G))

    return render_template(
        "career_path.html",
        roles=roles_sorted,
        selected_role=selected_role,
        suggestions=suggestions,
        missing_map=missing_map,
    )



if __name__ == "__main__":
    app.run(debug=True)
