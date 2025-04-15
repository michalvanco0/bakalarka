import pandas as pd
from pyvis.network import Network


def export_csv(G, filename):
    edge_list = []
    for source, target, attr in G.edges(data=True):
        edge_list.append({
            "Source": source,
            "Target": target,
        })

    df = pd.DataFrame(edge_list)

    df.to_csv(filename + ".csv", index=False)


def export_html(G, name):
    try:
        net = Network(notebook=False, height="750px", width="100%", bgcolor="#ffffff", font_color="black")
        net.from_nx(G)
        net.show(name + ".html", notebook=False)
    except Exception as e:
        print(f"An error occurred: {e}")


def save_graph(fig, file):
    fig.savefig(file, format('png'), dpi=300)