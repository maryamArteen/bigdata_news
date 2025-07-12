import rdflib
from rdflib.namespace import RDF
from pyvis.network import Network

# Load RDF graph
g = rdflib.Graph()
g.parse("articles_data2.ttl", format="turtle")
EX = rdflib.Namespace("http://example.org/misinfo#")

# Helper to label nodes
def label_node(node):
    if isinstance(node, rdflib.URIRef):
        return node.split('#')[-1]
    elif isinstance(node, rdflib.Literal):
        return str(node)
    else:
        return str(node)

# Helper to build pyvis graph from filtered RDF triples
def build_pyvis_graph(triples, title, color="lightblue"):
    net = Network(height="500px", width="100%", notebook=False)
    net.force_atlas_2based()

    net.heading = title  # This is just a note; we'll add HTML outside later

    for s, p, o in triples:
        s_label = label_node(s)
        o_label = label_node(o)
        p_label = label_node(p)

        # Add nodes with colors by type
        for node, lbl in [(s, s_label), (o, o_label)]:
            if not net.get_node(node):
                net.add_node(node, label=lbl, title=str(node), color=color)

        # Add edge with predicate as label
        net.add_edge(s, o, title=p_label)

    return net

# --- Filter functions ---
def filter_by_predicate(pred):
    return [(s,p,o) for s,p,o in g if p == pred]

# Build graphs
graphs = []

# 1. Articles -> Labels
graphs.append(("Articles and their Labels", filter_by_predicate(EX.hasLabel), "#97C2FC"))

# 2. Articles -> Subjects
graphs.append(("Articles and their Subjects", filter_by_predicate(EX.hasSubject), "#FFA07A"))

# 3. Articles -> Topics
graphs.append(("Articles and their Topics", filter_by_predicate(EX.hasTopic), "#90EE90"))

# 4. Articles -> Entities
graphs.append(("Articles and their Entities", filter_by_predicate(EX.hasEntity), "#DA70D6"))

# 5. Full integrated graph (excluding literals as nodes)
full_triples = [(s,p,o) for s,p,o in g if not isinstance(o, rdflib.Literal)]
graphs.append(("Full Knowledge Graph", full_triples, "#FFB347"))

# Create pyvis networks
nets = [build_pyvis_graph(triples, title, color) for title, triples, color in graphs]

# Save all networks into one HTML page using pyvis' save_html and simple HTML wrapper
html_parts = []
for idx, net in enumerate(nets):
    html_file = f"graph_{idx}.html"
    net.save_graph(html_file)

    # Extract just the <body> content from each saved html to embed
    with open(html_file, "r", encoding="utf-8") as f:
        content = f.read()
    body_start = content.find("<body>")
    body_end = content.find("</body>")
    body_content = content[body_start + 6 : body_end]

    # Add a section header and the graph
    html_parts.append(f"<h2>{graphs[idx][0]}</h2>\n{body_content}")

# Create the full HTML page with all graphs
full_html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <title>Knowledge Graph Visualizations</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h2 {{ border-bottom: 2px solid #444; padding-bottom: 5px; }}
        .graph-container {{ margin-bottom: 50px; }}
    </style>
</head>
<body>
    <h1>Knowledge Graph Visualizations</h1>
    {''.join(f'<div class="graph-container">{part}</div>' for part in html_parts)}
</body>
</html>
"""

# Save the combined file
with open("all_graphs_combined.html", "w", encoding="utf-8") as f:
    f.write(full_html)

print("All graphs saved to all_graphs_combined.html — open this in any modern browser.")