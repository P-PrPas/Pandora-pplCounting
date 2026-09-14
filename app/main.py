"""Entry point: load RTSP_URL from .env, validate the camera is reachable,
grab a snapshot for zone drawing, then launch the main window.
"""
import os
import sys
from pathlib import Path

import cv2
from dotenv import load_dotenv
from PySide6.QtWidgets import QApplication, QMessageBox

sys.path.insert(0, str(Path(__file__).parent))
from main_window import MainWindow

load_dotenv(Path(__file__).parent.parent / ".env")


def grab_snapshot(source):
    cap = cv2.VideoCapture(source)
    ok, frame = cap.read() if cap.isOpened() else (False, None)
    cap.release()
    return frame if ok else None


def main():
    app = QApplication(sys.argv)

    source = os.environ.get("RTSP_URL")
    if not source:
        QMessageBox.critical(None, "Missing configuration",
                              "RTSP_URL is not set.\n\nCopy .env.example to .env at the repo "
                              "root and set RTSP_URL to your camera's stream.")
        return 1

    snapshot = grab_snapshot(source)
    if snapshot is None:
        QMessageBox.critical(None, "Camera unreachable",
                              f"Could not read a frame from:\n{source}\n\n"
                              "Check the camera is powered on and the URL/credentials are correct, "
                              "then restart the app.")
        return 1

    window = MainWindow(source, snapshot)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
