from PyQt6.QtWidgets import QWidget, QPushButton, QLabel, QFileDialog, QVBoxLayout, QTextEdit, QButtonGroup, \
    QRadioButton, QLineEdit
from text_processing import preprocess_text
from network import build_network, save_graph, plot_networks, plot_logbining
from analysis import compare_networks, log_bin_degrees
import os
from config_setter import load_config, update_config

ALL_PUNCTUATION = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'


class GraphAnalysisApp(QWidget):
    def __init__(self):
        super().__init__()
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
        self.rb2 = QRadioButton('.,?!;', self)
        self.rb3 = QRadioButton("Vlastné", self)
        self.radio_group.addButton(self.rb1, 1)
        self.radio_group.addButton(self.rb2, 2)
        self.radio_group.addButton(self.rb3, 3)
        self.custom_input = QLineEdit(self)
        self.custom_input.setPlaceholderText("Vyber vlastnú interpunkciu")
        self.custom_input.setEnabled(False)
        self.radio_group.buttonClicked.connect(self.on_radio_change)

        self.btn_analyze = QPushButton("Analyzovať Text")
        self.btn_analyze.clicked.connect(self.analyze_text)

        self.results = QTextEdit()
        self.results.setReadOnly(True)

        self.btn_save = QPushButton("Uložiť Analýzu")
        self.btn_save.clicked.connect(self.save_analysis)

        layout = QVBoxLayout()
        layout.addWidget(self.rb1)
        layout.addWidget(self.rb2)
        layout.addWidget(self.rb3)
        layout.addWidget(self.custom_input)
        layout.addWidget(self.label)
        layout.addWidget(self.btn_select)
        layout.addWidget(self.btn_analyze)
        layout.addWidget(self.results)
        layout.addWidget(self.btn_save)
        self.setLayout(layout)

    def set_punctuation_pattern(self):
        if self.radio_group.checkedId() == 1:
            update_config("punctuation_pattern", ALL_PUNCTUATION)
        elif self.radio_group.checkedId() == 2:
            update_config("punctuation_pattern", ALL_PUNCTUATION)
        else:
            update_config("punctuation_pattern", self.custom_input.text())

    def on_radio_change(self):
        if self.radio_group.checkedId() == 3:
            self.custom_input.setEnabled(True)
        else:
            self.custom_input.setEnabled(False)

    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Otvoriť", "", "Text Files (*.txt)")
        if file_path:
            self.label.setText(f"{file_path}")
            self.file_path = file_path

    def analyze_text(self):
        if self.file_path == "Žiaden súbor":
            self.results.setText("Nie je vybratý súbor!")
            return
        if self.custom_input.text() == "" and self.radio_group.checkedId() == 3:
            self.results.setText("Vyber vlastnú interpunkciu!")
            return

        with open(self.file_path, "r", encoding="utf-8") as file:
            text = file.read()

        tokens_with = preprocess_text(text, keep_punctuation=True)
        tokens_without = preprocess_text(text, keep_punctuation=False)

        g_with = build_network(tokens_with)
        g_without = build_network(tokens_without)

        results = compare_networks(g_with, g_without)
        self.results.setText(results)

        self.G1, self.G2 = g_with, g_without
        plot_networks([self.G1, self.G2], ["Graf S interpunkciou", "Graf BEZ interpunkcie"])
        degrees_with = [deg for _, deg in self.G1.degree()]
        degrees_without = [deg for _, deg in self.G2.degree()]

        bin_centers_with, hist_with = log_bin_degrees(degrees_with)
        bin_centers_without, hist_without = log_bin_degrees(degrees_without)

        plot_logbining(bin_centers_with, bin_centers_without, hist_with, hist_without)

    def save_analysis(self):
        save_folder = QFileDialog.getExistingDirectory(self, "Vyber priečinok pre uloženie")
        if save_folder and hasattr(self, 'G1') and hasattr(self, 'G2'):
            save_graph(self.G1, os.path.join(save_folder, "withPunc.png"))
            save_graph(self.G2, os.path.join(save_folder, "withoutPunc.png"))
            with open(os.path.join(save_folder, "analysis.txt"), "w") as f:
                f.write(self.results.toPlainText())
