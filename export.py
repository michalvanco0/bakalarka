import tempfile
import pandas as pd
from matplotlib import pyplot as plt
from pyvis.network import Network
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle


def export_csv(G, filename):
    edge_list = []
    for source, target, attr in G.edges(data=True):
        edge_list.append({
            "Source": source,
            "Target": target,
        })

    df = pd.DataFrame(edge_list)
    df.to_csv(filename, index=False)


def export_html(G, filename):
    net = Network(notebook=False, height="750px", width="100%", bgcolor="#ffffff", font_color="black")
    net.barnes_hut()

    for node in G.nodes(data=True):
        node0 = node[0]
        label = str(node0)
        deg = G.degree(node0)
        color = "#e74c3c" if deg > 3 else "#3498db"
        net.add_node(node0, label=label, color=color, title=str(node0))
    for src, tgt, attrs in G.edges(data=True):
        net.add_edge(src, tgt, title=str(attrs))
    net.show_buttons(filter_=['physics'])
    net.show(filename, notebook=False)


def export_pdf(name, results_table=None, results_table_2=None, graphs=[]):

    pdf_path = name if name.endswith('.pdf') else name + '.pdf'

    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = [Paragraph("Network Analysis", styles['Heading1']), Spacer(1, 12)]

    if results_table:
        elements.append(Paragraph("Comparison of Networks", styles['Heading2']))
        table = Table(results_table, colWidths=[140, 180, 180])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (1, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ]))
        elements.append(table)
        elements.append(Spacer(1, 12))

    if results_table_2:
        stats_table_obj = Table(results_table_2, colWidths=[180, 320])
        stats_table_obj.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (1, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ]))
        elements.append(PageBreak())
        elements.append(Paragraph("Textual Statistics", styles['Heading2']))
        elements.append(stats_table_obj)

    def fig_to_image(fig):
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        fig.savefig(tmp_file.name, bbox_inches='tight')
        plt.close(fig)
        return Image(tmp_file.name, width=5.5 * inch, height=4.2 * inch)

    for i, fig in enumerate(graphs):
        elements.append(PageBreak())
        # elements.append(Paragraph(f"Network {i + 1}", styles['Heading2']))
        elements.append(fig_to_image(fig))

    doc.build(elements)
