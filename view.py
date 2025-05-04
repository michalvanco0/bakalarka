from PyQt6.QtWidgets import QWidget, QPushButton, QLabel, QFileDialog, QVBoxLayout, QTextEdit, QButtonGroup, \
    QRadioButton, QLineEdit, QCheckBox, QLabel, QHBoxLayout, QSlider
from PyQt6.QtCore import Qt, QTimer, QThread
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph

import network
from text_processing import tokenize_text
from network import build_network, plot_logbining, plot_networks
from analysis import compare_networks, ling_analysis
from export import export_csv, export_html, export_pdf
from config_setter import update_config, load_config

ALL_PUNCTUATION = '!"\'()*+,-./:;<=>?[]^_`{|}~»«'
BASIC = '.?!,'


class GraphAnalysisApp(QWidget):
    def __init__(self):
        super().__init__()
        self.table = None
        self.show_fitconvergence_checkbox = None
        self.show_histogram_checkbox = None
        self.degree_value = None
        self.degree_slider = None
        self.filter_nodes_checkbox = None
        self.show_net_checkbox = None
        self.label_checkbox = None
        self.directed_checkbox = None
        self.weighted_checkbox = None
        self.plotted_graphs = {}
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
        self.file_path = "Žiaden súbor"
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Punctuation Network Analyzer")
        self.setGeometry(100, 100, 800, 600)

        self.label = QLabel("Vyber súbor:")
        self.btn_select = QPushButton("Prehľadávať")
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
        self.rb3 = QRadioButton("Vlastné", self)
        self.radio_group.addButton(self.rb1, 1)
        self.radio_group.addButton(self.rb2, 2)
        self.radio_group.addButton(self.rb3, 3)
        self.custom_input = QLineEdit(self)
        self.custom_input.setPlaceholderText("Vyber vlastnú interpunkciu")
        self.custom_input.setEnabled(False)
        self.radio_group.buttonClicked.connect(self.on_radio_change)

        self.weighted_checkbox = QCheckBox("Vážnený", self)
        self.directed_checkbox = QCheckBox("Orientovaný", self)
        self.label_checkbox = QCheckBox("Zobraziť slová v sieti", self)
        self.show_net_checkbox = QCheckBox("Zobraziť slovnu sieť", self)
        self.show_histogram_checkbox = QCheckBox("Zobraziť histogram", self)
        self.show_fitconvergence_checkbox = QCheckBox("Zobraziť fit konvergenciu", self)
        self.degree_slider = QSlider(Qt.Orientation.Horizontal)
        self.degree_slider.setMinimum(0)
        self.degree_slider.setMaximum(100)
        self.degree_slider.setValue(0)
        self.degree_value = QLabel(f'Min. stupeň: {self.degree_slider.value()}')
        self.degree_slider.valueChanged.connect(self.update_degree_value)

        self.btn_analyze = QPushButton("Analyzovať Text")
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

        self.btn_stop = QPushButton("Zastaviť analýzu")
        self.btn_stop.clicked.connect(self.stop_analysis)
        self.btn_stop.setEnabled(False)

        self.results = QTextEdit()
        self.results.setReadOnly(True)

        self.btn_export_csv_G1 = QPushButton("Exportovať G1 do CSV")
        self.btn_export_csv_G1.clicked.connect(lambda: self.export_csv_dialog(self.G1))

        self.btn_export_csv_G2 = QPushButton("Exportovať G2 do CSV")
        self.btn_export_csv_G2.clicked.connect(lambda: self.export_csv_dialog(self.G2))

        self.btn_export_html_G1 = QPushButton("Exportovať G1 do HTML")
        self.btn_export_html_G1.clicked.connect(lambda: self.export_html_dialog(self.G1))

        self.btn_export_html_G2 = QPushButton("Exportovať G2 do HTML")
        self.btn_export_html_G2.clicked.connect(lambda: self.export_html_dialog(self.G2))

        self.btn_save = QPushButton("Uložiť Analýzu")
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
        layout1.addWidget(self.label)
        layout1.addWidget(self.btn_select)
        layout2.addWidget(self.rb1)
        layout2.addWidget(self.rb2)
        layout2.addWidget(self.rb3)
        layout2.addWidget(self.custom_input)
        layout3.addWidget(self.weighted_checkbox)
        layout3.addWidget(self.directed_checkbox)
        layout3.addWidget(self.label_checkbox)
        layout3.addWidget(self.show_net_checkbox)
        layout3.addWidget(self.show_histogram_checkbox)
        layout3.addWidget(self.show_fitconvergence_checkbox)
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
        layout1.addWidget(self.btn_stop)
        layout.addLayout(layout1)
        layout.addWidget(self.results)
        self.setLayout(layout)

    def update_degree_value(self):
        self.degree_value.setText(f'Min. stupeň: {self.degree_slider.value()}')

    def set_configuration(self):
        if self.radio_group.checkedId() == 1:
            update_config("punctuation_pattern", ALL_PUNCTUATION)
        elif self.radio_group.checkedId() == 2:
            update_config("punctuation_pattern", BASIC)
        else:
            update_config("punctuation_pattern", self.custom_input.text())

        update_config("directed", self.directed_checkbox.isChecked())
        update_config("weighted", self.weighted_checkbox.isChecked())
        update_config("show_labels", self.label_checkbox.isChecked())
        update_config("show_net", self.show_net_checkbox.isChecked())
        update_config("show_histogram", self.show_histogram_checkbox.isChecked())
        update_config("show_fit_convergence", self.show_fitconvergence_checkbox.isChecked())
        update_config("min_degree", self.degree_slider.value())

    def on_radio_change(self):
        self.custom_input.setEnabled(self.radio_group.checkedId() == 3)

    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Otvoriť", "", "Text Files (*.txt)")
        if file_path:
            self.label.setText(f"{file_path}")
            self.file_path = file_path

    def analyze_text(self):
        self.set_configuration()
        config = load_config()
        if self.file_path == "Žiaden súbor":
            self.show_snackbar("Nie je vybratý súbor!")
            return
        if (self.custom_input.text() == "" or self.custom_input.text() is None) and self.radio_group.checkedId() == 3:
            self.show_snackbar("Nevybral si interpunkciu! Analýza bez interpunkcie.")

        with open(self.file_path, "r", encoding="utf-8") as file:
            text = file.read()

        tokens_with = tokenize_text(text, keep_punctuation=True)
        tokens_without = tokenize_text(text, keep_punctuation=False)

        self.G1 = build_network(tokens_with)
        self.G2 = build_network(tokens_without)

        self.table = compare_networks(self.G1, self.G2)
        results = ling_analysis(text)
        self.results.setText(results)

        if config["show_net"]:
            self.plotted_graphs["wordnets"] = plot_networks([self.G1, self.G2],
                                                            ["Sieť S interpunkciou", "Sieť BEZ interpunkcie"])
        if config["show_histogram"]:
            self.plotted_graphs["histogram"] = plot_logbining(self.G1, self.G2)
        if config["show_fit_convergence"]:
            self.plotted_graphs["fit_convergence"] = (
                network.plot_fit_convergence(network.fit_convergence_analysis(text, step_size=1000)))
        distances = network.get_punctuation_distances(tokens_with, set(config["punctuation_pattern"]))
        ks, freqs, q_fit, beta_fit = network.fit_weibull_like_model(distances)

        if q_fit and beta_fit:
            print(f"\n--- Weibull-like Distribúcia (interpunkcie) ---\n")
            print(f"q (frekvencia): {q_fit:.4f}\nβ (tvar): {beta_fit:.4f}\n")
            self.plotted_graphs["weibull_like"] = network.plot_weibull_like_fit(ks, freqs, q_fit, beta_fit)
        else:
            print("\nNepodarilo sa prispôsobiť Weibull-like distribúciu.\n")

    def stop_analysis(self):
        self._stop_analysis = True
        self.show_snackbar("Analýza bola zastavená")

    def save_analysis(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save PDF Report",
            "report.pdf",
            "PDF Files (*.pdf)"
        )
        try:
            table = prepare_table_for_pdf(self.table)
            export_pdf(file_path, table, self.results.toPlainText(), self.plotted_graphs["histogram"])
            self.show_snackbar("Uložené do PDF")
        except Exception as e:
            print(f"An error occurred while saving the analysis: {e}")
            self.show_snackbar("Chyba pri ukladaní do PDF")

    def export_csv_dialog(self, graph):
        path, _ = QFileDialog.getSaveFileName(self, "Export CSV", "graph.csv", "CSV Files (*.csv)")
        if path:
            try:
                export_csv(graph, path)
                self.show_snackbar("Exportované do CSV")
            except Exception as e:
                self.show_snackbar("Chyba pri exportovaní do CSV")

    def export_html_dialog(self, graph):
        path, _ = QFileDialog.getSaveFileName(self, "Export HTML", "graph.html", "HTML Files (*.html)")
        if path:
            try:
                export_html(graph, path)
                self.show_snackbar("Exportované do HTML")
            except Exception as e:
                self.show_snackbar("Chyba pri exportovaní do HTML")

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


def table_to_html(table_data):
    html = '<table border="1" cellspacing="0" cellpadding="4" style="border-collapse: collapse;">'
    for i, row in enumerate(table_data):
        html += "<tr>"
        for cell in row:
            tag = "th" if i == 0 else "td"
            html += f"<{tag}>{cell}</{tag}>"
        html += "</tr>"
    html += "</table>"
    return html


def prepare_table_for_pdf(table_data):
    styles = getSampleStyleSheet()
    normal_style = styles["Normal"]

    wrapped_data = []
    for row in table_data:
        wrapped_row = [Paragraph(str(cell), normal_style) for cell in row]
        wrapped_data.append(wrapped_row)
    return wrapped_data