from PyQt6.QtWidgets import QApplication
from view import GraphAnalysisApp
import sys

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GraphAnalysisApp()
    window.show()
    sys.exit(app.exec())
