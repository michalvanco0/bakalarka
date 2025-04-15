from PyQt6.QtCore import pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import QWidget, QPushButton, QLabel, QFileDialog, QVBoxLayout, QTextEdit, QButtonGroup, \
    QRadioButton, QLineEdit, QHBoxLayout, QMessageBox, QDialog, QCheckBox

import network
from text_processing import tokenize_text
from network import build_network, plot_networks, plot_logbining
from analysis import compare_networks, ling_analysis
from export import save_graph, export_csv, export_html
import os
from config_setter import load_config, update_config

ALL_PUNCTUATION = '!"\'()*+,-./:;<=>?[]^_`{|}~»«'
BASIC = '.?!,'


class GraphAnalysisApp(QWidget):
    def __init__(self):
        super().__init__()
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

        self.btn_analyze = QPushButton("Analyzovať Text")
        self.btn_analyze.clicked.connect(self.analyze_text)

        self.results = QTextEdit()
        self.results.setReadOnly(True)

        self.btn_export_csv_G1 = QPushButton("Exportovať G1 do CSV")
        self.btn_export_csv_G1.clicked.connect(lambda: export_csv(self.G1, "G1"))

        self.btn_export_csv_G2 = QPushButton("Exportovať G2 do CSV")
        self.btn_export_csv_G2.clicked.connect(lambda: export_csv(self.G2, "G2"))

        self.btn_export_html_G1 = QPushButton("Exportovať G1 do HTML")
        self.btn_export_html_G1.clicked.connect(lambda: export_html(self.G1, "G1"))

        self.btn_export_html_G2 = QPushButton("Exportovať G2 do HTML")
        self.btn_export_html_G2.clicked.connect(lambda: export_html(self.G2, "G2"))

        self.btn_save = QPushButton("Uložiť Analýzu")
        self.btn_save.clicked.connect(self.save_analysis)

        layout = QVBoxLayout()
        layout.addWidget(self.rb1)
        layout.addWidget(self.rb2)
        layout.addWidget(self.rb3)
        layout.addWidget(self.custom_input)
        layout.addWidget(self.label)
        layout.addWidget(self.btn_select)
        layout.addWidget(self.weighted_checkbox)
        layout.addWidget(self.directed_checkbox)
        layout.addWidget(self.label_checkbox)
        layout.addWidget(self.btn_analyze)
        layout.addWidget(self.results)
        layout.addWidget(self.btn_export_csv_G1)
        layout.addWidget(self.btn_export_csv_G2)
        layout.addWidget(self.btn_export_html_G1)
        layout.addWidget(self.btn_export_html_G2)
        layout.addWidget(self.btn_save)
        self.setLayout(layout)

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

    def on_radio_change(self):
        self.custom_input.setEnabled(self.radio_group.checkedId() == 3)

    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Otvoriť", "", "Text Files (*.txt)")
        if file_path:
            self.label.setText(f"{file_path}")
            self.file_path = file_path

    def analyze_text(self):
        self.set_configuration()
        if self.file_path == "Žiaden súbor":
            self.results.setText("Nie je vybratý súbor!")
            return
        if self.custom_input.text() == "" and self.radio_group.checkedId() == 3:
            self.results.setText("Vyber vlastnú interpunkciu!")
            return

        with open(self.file_path, "r", encoding="utf-8") as file:
            text = file.read()

        tokens_with = tokenize_text(text, keep_punctuation=True)
        tokens_without = tokenize_text(text, keep_punctuation=False)

        self.G1 = build_network(tokens_with)
        self.G2 = build_network(tokens_without)
        # self.plotted_graphs["wordnets"] = plot_networks([self.G1, self.G2],
        #                                                 ["Sieť S interpunkciou", "Sieť BEZ interpunkcie"])

        results = compare_networks(self.G1, self.G2)
        results += ling_analysis(text)
        self.results.setText(results)

        g = network.dorogov_model(len(self.G1.nodes())*5)
        self.plotted_graphs["histogram"] = plot_logbining(self.G1, self.G2)

        # export_html(self.G1, "G1")
        # export_html(self.G2, "G2")
        # export_html(g, "g")

    def save_analysis(self):
        save_folder = QFileDialog.getExistingDirectory(self, "Vyber priečinok pre uloženie")
        if save_folder and hasattr(self, 'G1') and hasattr(self, 'G2'):
            save_graph(self.G1, os.path.join(save_folder, "withPunc.png"))
            save_graph(self.G2, os.path.join(save_folder, "withoutPunc.png"))
            with open(os.path.join(save_folder, "analysis.txt"), "w") as f:
                f.write(self.results.toPlainText())
