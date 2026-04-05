import os
from datetime import datetime

import mss


CAPTURE_DIR = "captures"


def ensure_capture_dir():
    os.makedirs(CAPTURE_DIR, exist_ok=True)


def capture_full_screen():
    ensure_capture_dir()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(CAPTURE_DIR, f"capture_{timestamp}.png")

    with mss.mss() as sct:
        monitor = sct.monitors[1]  # primary monitor
        sct_img = sct.grab(monitor)
        mss.tools.to_png(sct_img.rgb, sct_img.size, output=output_path)

    return output_path