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


def export_pdf(name, results_table=None, results="", histo_fig=None, G1_fig=None, G2_fig=None, extra_figs=None):
    if extra_figs is None:
        extra_figs = []

    pdf_path = name if name.endswith('.pdf') else name + '.pdf'

    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

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

    paragraphs = results.strip().split("\n")
    for p in paragraphs:
        elements.append(Paragraph(p.strip().replace("  ", "&nbsp; "), styles['Normal']))
        elements.append(Spacer(1, 6))

    def fig_to_image(fig):
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        fig.savefig(tmp_file.name, bbox_inches='tight')
        plt.close(fig)
        return Image(tmp_file.name, width=5.5 * inch, height=4.2 * inch)

    # Add figures
    if histo_fig:
        elements.append(PageBreak())
        elements.append(Paragraph("Histogram", styles['Heading2']))
        elements.append(fig_to_image(histo_fig))

    if G1_fig:
        elements.append(PageBreak())
        elements.append(Paragraph("Sieť S interpunkciou (G1)", styles['Heading2']))
        elements.append(fig_to_image(G1_fig))

    if G2_fig:
        elements.append(PageBreak())
        elements.append(Paragraph("Sieť BEZ interpunkcie (G2)", styles['Heading2']))
        elements.append(fig_to_image(G2_fig))

    for i, fig in enumerate(extra_figs):
        elements.append(PageBreak())
        elements.append(Paragraph(f"Extra graf {i + 1}", styles['Heading2']))
        elements.append(fig_to_image(fig))

    doc.build(elements)
