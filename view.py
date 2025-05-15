from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QWidget, QPushButton, QLabel, QFileDialog, QVBoxLayout, QTextEdit, QButtonGroup, \
    QRadioButton, QLineEdit, QCheckBox, QLabel, QHBoxLayout, QSlider
from PyQt6.QtCore import Qt, QTimer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph

import network
from text_processing import tokenize_text
from network import build_network, plot_digree_distribution, plot_networks
from analysis import compare_networks, ling_analysis, get_weibull_parameters
from export import export_csv, export_html, export_pdf
from config_setter import update_config, load_config, ALL_PUNCTUATION, BASIC


class GraphAnalysisApp(QWidget):
    def __init__(self):
        super().__init__()
        self.show_binned_checkbox = None
        self.model_window = None
        self.show_models_checkbox = None
        self.show_distribution_comparison_checkbox = None
        self.pict_label = None
        self.image = None
        self.show_weibull_checkbox = None
        self.table = None
        self.table_2 = None
        self.show_fit_convergence_checkbox = None
        self.show_histogram_checkbox = None
        self.degree_value = None
        self.degree_slider = None
        self.filter_nodes_checkbox = None
        self.show_net_checkbox = None
        self.label_checkbox = None
        self.plotted_graphs = []
        self.btn_save = None
        self.btn_analyze = None
        self.custom_input = None
        self.rb3 = None
        self.rb2 = None
        self.rb1 = None
        self.radio_group = None
        self.btn_select = None
        self.label = None
        self.results = None
        self.G2 = None
        self.G1 = None
        self.file_path = "No file"
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Punctuation Network Analyzer")
        self.setGeometry(100, 100, 800, 600)
        self.setWindowIcon(QIcon("logo1.png"))
        self.label = QLabel("SELECT FILE")
        self.btn_select = QPushButton("SEARCH FILES")
        self.btn_select.clicked.connect(self.select_file)
        self.btn_select.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50; 
                color: white; 
                padding: 10px 20px; 
                border-radius: 5px; 
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)

        self.radio_group = QButtonGroup(self)
        self.rb1 = QRadioButton(ALL_PUNCTUATION, self)
        self.rb1.setChecked(True)
        self.rb2 = QRadioButton(BASIC, self)
        self.rb3 = QRadioButton("Custom", self)
        self.radio_group.addButton(self.rb1, 1)
        self.radio_group.addButton(self.rb2, 2)
        self.radio_group.addButton(self.rb3, 3)
        self.custom_input = QLineEdit(self)
        self.custom_input.setPlaceholderText("...")
        self.custom_input.setEnabled(False)
        self.radio_group.buttonClicked.connect(self.on_radio_change)

        self.label_checkbox = QCheckBox("Words in the network", self)
        self.show_net_checkbox = QCheckBox("Word network", self)
        self.show_histogram_checkbox = QCheckBox("Degree distribution", self)
        self.show_binned_checkbox = QCheckBox("Binned degree distribution", self)
        self.show_models_checkbox = QCheckBox("Models", self)
        self.show_fit_convergence_checkbox = QCheckBox("Fit convergence", self)
        self.show_weibull_checkbox = QCheckBox("Weibull Distribution", self)
        self.show_distribution_comparison_checkbox = QCheckBox("Distribution comparison", self)
        self.degree_slider = QSlider(Qt.Orientation.Horizontal)
        self.degree_slider.setMinimum(0)
        self.degree_slider.setMaximum(100)
        self.degree_slider.setValue(0)
        self.degree_value = QLabel(f'Min. degree: {self.degree_slider.value()}')
        self.degree_slider.valueChanged.connect(self.update_degree_value)

        self.btn_analyze = QPushButton("Analyze text")
        self.btn_analyze.clicked.connect(self.analyze_text)
        self.btn_analyze.setStyleSheet("""
            QPushButton {
               background-color: #4CAF50;
               color: white;
               padding: 10px 20px;
               border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
               """)

        self.results = QTextEdit()
        self.results.setReadOnly(True)

        self.btn_export_csv_G1 = QPushButton("Export G1 into CSV")
        self.btn_export_csv_G1.clicked.connect(lambda: self.export_csv_dialog(self.G1))

        self.btn_export_csv_G2 = QPushButton("Export G2 into CSV")
        self.btn_export_csv_G2.clicked.connect(lambda: self.export_csv_dialog(self.G2))

        self.btn_export_html_G1 = QPushButton("Export G1 into HTML")
        self.btn_export_html_G1.clicked.connect(lambda: self.export_html_dialog(self.G1))

        self.btn_export_html_G2 = QPushButton("Export G2 into HTML")
        self.btn_export_html_G2.clicked.connect(lambda: self.export_html_dialog(self.G2))

        self.btn_save = QPushButton("Save analysis")
        self.btn_save.clicked.connect(self.save_analysis)
        self.btn_save.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50; 
                color: white; 
                padding: 10px 20px; 
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)

        layout = QHBoxLayout()
        layout1 = QVBoxLayout()
        layout2 = QVBoxLayout()
        layout3 = QVBoxLayout()
        layout4 = QHBoxLayout()
        layout1.addWidget(self.pict_label)
        layout1.addWidget(self.label)
        layout1.addWidget(self.btn_select)
        layout2.addWidget(self.rb1)
        layout2.addWidget(self.rb2)
        layout2.addWidget(self.rb3)
        layout2.addWidget(self.custom_input)
        layout3.addWidget(self.label_checkbox)
        layout3.addWidget(self.show_net_checkbox)
        layout3.addWidget(self.show_histogram_checkbox)
        layout3.addWidget(self.show_binned_checkbox)
        layout3.addWidget(self.show_models_checkbox)
        layout3.addWidget(self.show_fit_convergence_checkbox)
        layout3.addWidget(self.show_weibull_checkbox)
        layout3.addWidget(self.show_distribution_comparison_checkbox)
        layout3.addWidget(self.degree_slider)
        layout3.addWidget(self.degree_value)
        layout4.addLayout(layout2)
        layout4.addLayout(layout3)
        layout1.addLayout(layout4)
        layout1.addWidget(self.btn_analyze)
        layout1.addWidget(self.btn_export_csv_G1)
        layout1.addWidget(self.btn_export_csv_G2)
        layout1.addWidget(self.btn_export_html_G1)
        layout1.addWidget(self.btn_export_html_G2)
        layout1.addWidget(self.btn_save)
        layout.addLayout(layout1)
        layout.addWidget(self.results)
        self.setLayout(layout)

    def update_degree_value(self):
        self.degree_value.setText(f'Min. degree: {self.degree_slider.value()}')

    def set_configuration(self):
        if self.radio_group.checkedId() == 1:
            update_config("punctuation_pattern", ALL_PUNCTUATION)
        elif self.radio_group.checkedId() == 2:
            update_config("punctuation_pattern", BASIC)
        else:
            update_config("punctuation_pattern", self.custom_input.text())

        update_config("show_labels", self.label_checkbox.isChecked())
        update_config("show_net", self.show_net_checkbox.isChecked())
        update_config("show_histogram", self.show_histogram_checkbox.isChecked())
        update_config("show_binned", self.show_binned_checkbox.isChecked())
        update_config("show_models", self.show_models_checkbox.isChecked())
        update_config("show_fit_convergence", self.show_fit_convergence_checkbox.isChecked())
        update_config("min_degree", self.degree_slider.value())
        update_config("show_weibull", self.show_weibull_checkbox.isChecked())
        update_config("show_distribution_comparison", self.show_distribution_comparison_checkbox.isChecked())

    def on_radio_change(self):
        self.custom_input.setEnabled(self.radio_group.checkedId() == 3)

    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open", "", "Text Files (*.txt)")
        if file_path:
            self.label.setText(f"{file_path}")
            self.file_path = file_path

    def analyze_text(self):
        self.show_snackbar("Analyzing...")
        self.plotted_graphs = []
        try:
            self.set_configuration()
            config = load_config()
            if self.file_path == "No file":
                self.show_snackbar("No file chosen!")
                return
            if ((self.custom_input.text() == "" or self.custom_input.text() is None) and
                    self.radio_group.checkedId() == 3):
                self.show_snackbar("Ignoring punctuation!")

            with open(self.file_path, "r", encoding="utf-8") as file:
                text = file.read()

            tokens_with = tokenize_text(text, keep_punctuation=True)
            tokens_without = tokenize_text(text, keep_punctuation=False)

            self.G1 = build_network(tokens_with)
            self.G2 = build_network(tokens_without)

            count_nodes = len(self.G1.nodes)
            models = network.generate_models(count_nodes)
            models["empirical G1"] = self.G1
            slopes = network.compute_slopes(models)
            self.table = compare_networks(self.G1, self.G2)
            self.table_2 = ling_analysis(text, slopes)

            ks, freqs, q, beta = get_weibull_parameters(tokens_with, config["punctuation_pattern"])

            if config["show_net"]:
                self.plotted_graphs.append(plot_networks(
                    [self.G1, self.G2],["Network with punctuation", "Network without punctuation"]))
            if config["show_binned"]:
                self.plotted_graphs.append(plot_digree_distribution(self.G1, self.G2))
                self.plotted_graphs.append(plot_digree_distribution(self.G1, self.G2, binned=True,
                                                                    xscale="linear", yscale="log"))
                self.plotted_graphs.append(plot_digree_distribution(self.G1, self.G2, binned=True,
                                                                    xscale="linear", yscale="linear"))
            if config["show_histogram"]:
                self.plotted_graphs.append(plot_digree_distribution(self.G1, self.G2, binned=False,
                                                                    xscale="log", yscale="log"))
                self.plotted_graphs.append(plot_digree_distribution(self.G1, self.G2, binned=False,
                                                                    xscale="linear", yscale="log"))
                self.plotted_graphs.append(plot_digree_distribution(self.G1, self.G2, binned=False,
                                                                    xscale="linear", yscale="linear"))
            if config["show_models"]:
                self.model_window = network.show_model_plot_window(models)
                # self.plotted_graphs.append(self.model_window)
            if config["show_fit_convergence"]:
                self.plotted_graphs.append(network.plot_fit_convergence(
                    network.fit_convergence_analysis(text, step_size=1000)))
            if config["show_weibull"] and q and beta:
                self.plotted_graphs.append(network.plot_weibull_fit(ks, freqs, q, beta))
            if config["show_distribution_comparison"]:
                self.plotted_graphs.append(network.plot_distribution_comparisons(ks, freqs, q, beta))

            self.results.setHtml(table_to_html(self.table, self.table_2))
        except Exception as e:
            print(f"An error occurred during analysis: {e}")
            self.show_snackbar("Analysis unsuccessful")
            return
        self.show_snackbar("Analysis finished")

    def save_analysis(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save PDF Report",
            "report.pdf",
            "PDF Files (*.pdf)"
        )
        try:
            table = table_for_pdf(self.table)
            table2 = table_for_pdf(self.table_2)
            export_pdf(file_path, table, table2, self.plotted_graphs)
            self.show_snackbar("Saved PDF")
        except Exception as e:
            print(f"An error occurred while saving the analysis: {e}")
            self.show_snackbar("Error during saving into PDF")

    def export_csv_dialog(self, graph):
        path, _ = QFileDialog.getSaveFileName(self, "Export CSV", "graph.csv", "CSV Files (*.csv)")
        if path:
            try:
                export_csv(graph, path)
                self.show_snackbar("CSV export successful")
            except Exception as e:
                self.show_snackbar("Error during CSV export")

    def export_html_dialog(self, graph):
        path, _ = QFileDialog.getSaveFileName(self, "Export HTML", "graph.html", "HTML Files (*.html)")
        if path:
            try:
                export_html(graph, path)
                self.show_snackbar("HTML export successful")
            except Exception as e:
                self.show_snackbar("Error during HTML export")

    def show_snackbar(self, message, duration=2500):
        snackbar = QLabel(text=message, parent=self)
        snackbar.setStyleSheet("""
            background-color: #323232;
            color: white;
            padding: 10px 20px;
            border-radius: 8px;
            font-size: 13px;
        """)
        snackbar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        snackbar.adjustSize()

        x = (self.width() - snackbar.width()) // 4
        y = self.height() - snackbar.height() - 30
        snackbar.move(x, y)
        snackbar.show()

        QTimer.singleShot(duration, snackbar.close)


def table_to_html(*table_datas):
    html = '''
    <style>
    table { border-collapse: collapse; width: 100%; }
    th, td { 
        border: 1px solid #ccc; 
        padding: 6px; 
        text-align: left;
        vertical-align: top;
        word-wrap: break-word;
        max-width: 300px;
    }
    </style>
    <table>
    '''
    for table_data in table_datas:
        for i, row in enumerate(table_data):
            html += "<tr>"
            for cell in row:
                tag = "th" if i == 0 else "td"
                html += f"<{tag}>{cell}</{tag}>"
            html += "</tr>"
        html += "</table>"
        html += "<br>"
    return html


def table_for_pdf(table_data):
    styles = getSampleStyleSheet()
    normal_style = styles["Normal"]

    wrapped_data = []
    for row in table_data:
        wrapped_row = [Paragraph(str(cell), normal_style) for cell in row]
        wrapped_data.append(wrapped_row)
    return wrapped_data
