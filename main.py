from PyQt6.QtWidgets import QApplication
from view import GraphAnalysisApp
import sys


def main():
    app = QApplication(sys.argv)
    window = GraphAnalysisApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
