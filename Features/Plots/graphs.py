import networkx as nx
import matplotlib.pyplot as plt

# Create a directed graph
G = nx.DiGraph()

# Adding nodes (user query, assistant response, incidents)
G.add_node("Q1", type="User Query", content="How can I resolve slow server issues?")
G.add_node("R1", type="Response", content="Have you tried restarting the server?")
G.add_node("I1", type="Incident", content="Incident 1234 - Server performance issues")
G.add_node("T1", type="Topic", content="Server Issues")

# Adding edges to represent relationships
G.add_edge("Q1", "R1", relationship="response")
G.add_edge("Q1", "I1", relationship="related_incident")
G.add_edge("Q1", "T1", relationship="related_topic")

# Draw the graph
plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G)  # Layout for better visualization
nx.draw(G, pos, with_labels=True, node_size=3000, node_color='lightblue', font_size=10)
labels = nx.get_edge_attributes(G, 'relationship')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
plt.show()