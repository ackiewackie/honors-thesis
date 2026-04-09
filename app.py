import tkinter as tk
from tkinter import scrolledtext, messagebox
import time

from pipeline.model_utils import load_model, classify_text, interpret_risk
from pipeline.ocr_utils import extract_text_from_image
from pipeline.screen_capture import capture_full_screen


BG      = "#1e1e2e"
BG_CARD = "#2a2a3d"
ACCENT  = "#7c6af7"
ACCENT_H= "#9d8fff"
FG      = "#cdd6f4"
FG_DIM  = "#6c7086"
SUCCESS = "#a6e3a1"
WARNING = "#f9e2af"
DANGER  = "#f38ba8"
FONT    = "Segoe UI"


class PhishingScannerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sightline: Real-time Protection")
        self.root.geometry("480x720")
        self.root.configure(bg=BG)
        self.root.resizable(False, False)

        self.model, self.tfidf, self.model_name = load_model()
        self.auto_scan_job = None
        self.last_capture_path = None
        self._last_risk_band = "LOW"

        self.build_ui()

    def _card(self, parent, **kwargs):
        return tk.Frame(parent, bg=BG_CARD, bd=0, relief="flat", **kwargs)

    def _label(self, parent, text, size=11, bold=False, dim=False, **kwargs):
        return tk.Label(parent, text=text,
                        font=(FONT, size, "bold" if bold else "normal"),
                        bg=parent["bg"], fg=FG_DIM if dim else FG, **kwargs)

    def _btn(self, parent, text, command, accent=True):
        bg = ACCENT if accent else BG_CARD
        btn = tk.Button(parent, text=text, command=command,
                        font=(FONT, 10, "bold"), bg=bg, fg="#ffffff",
                        activebackground=ACCENT_H, activeforeground="#ffffff",
                        relief="flat", bd=0, cursor="hand2", padx=16, pady=10)
        btn.bind("<Enter>", lambda e: btn.config(bg=ACCENT_H))
        btn.bind("<Leave>", lambda e: btn.config(bg=ACCENT if accent else BG_CARD))
        return btn

    def build_ui(self):
        # Header
        header = tk.Frame(self.root, bg=BG_CARD, pady=18)
        header.pack(fill="x")
        self._label(header, "Sightline", size=15, bold=True).pack()
        #self._label(header, f"Model: {self.model_name}", size=8, dim=True).pack(pady=(2, 0))

        # Buttons
        btn_frame = tk.Frame(self.root, bg=BG, pady=16)
        btn_frame.pack(fill="x", padx=24)
        self.scan_button = self._btn(btn_frame, "\u27f3  Scan Screen", self.scan_screen)
        self.scan_button.pack(fill="x", pady=(0, 8))
        self.auto_button = self._btn(btn_frame, "\u25b6  Start Auto Scan (3s)", self.toggle_auto_scan, accent=False)
        self.auto_button.pack(fill="x")

        # Status
        status_card = self._card(self.root, pady=10)
        status_card.pack(fill="x", padx=24, pady=(8, 0))
        self.status_label = self._label(status_card, "\u25cf  Ready", size=10, dim=True)
        self.status_label.pack()

        # Risk card
        risk_card = self._card(self.root, pady=16)
        risk_card.pack(fill="x", padx=24, pady=12)
        row = tk.Frame(risk_card, bg=BG_CARD)
        row.pack()

        score_col = tk.Frame(row, bg=BG_CARD, padx=24)
        score_col.pack(side="left")
        self._label(score_col, "RISK SCORE", size=8, dim=True).pack()
        self.score_label = self._label(score_col, "\u2014", size=22, bold=True)
        self.score_label.pack()

        tk.Frame(row, bg=FG_DIM, width=1, height=50).pack(side="left", fill="y", pady=4)

        band_col = tk.Frame(row, bg=BG_CARD, padx=24)
        band_col.pack(side="left")
        self._label(band_col, "RISK BAND", size=8, dim=True).pack()
        self.band_label = self._label(band_col, "\u2014", size=22, bold=True)
        self.band_label.pack()

        # Capture path
        path_card = self._card(self.root, padx=14, pady=10)
        path_card.pack(fill="x", padx=24, pady=(0, 8))
        self._label(path_card, "LAST CAPTURE", size=8, dim=True).pack(anchor="w")
        self.capture_path_text = self._label(path_card, "None", size=9, dim=True)
        self.capture_path_text.config(wraplength=420, justify="left")
        self.capture_path_text.pack(anchor="w", pady=(2, 0))

        # OCR text
        ocr_card = self._card(self.root, padx=14, pady=10)
        ocr_card.pack(fill="both", expand=True, padx=24, pady=(0, 16))
        self._label(ocr_card, "EXTRACTED TEXT", size=8, dim=True).pack(anchor="w")
        self.ocr_textbox = scrolledtext.ScrolledText(
            ocr_card, wrap=tk.WORD, font=(FONT, 9),
            bg="#13131f", fg=FG, insertbackground=FG,
            relief="flat", bd=0, padx=8, pady=8, selectbackground=ACCENT
        )
        self.ocr_textbox.pack(fill="both", expand=True, pady=(6, 0))
        self.ocr_textbox.tag_config("highlight", background=ACCENT, foreground="#ffffff")

    def set_risk_color(self, band):
        color = DANGER if band == "HIGH" else WARNING if band == "MEDIUM" else SUCCESS
        self.band_label.config(fg=color)
        self.score_label.config(fg=color)

    def scan_screen(self, auto=False):
        try:
            self.status_label.config(text="\u25cf  Preparing capture...", fg=FG_DIM)
            self.root.update_idletasks()

            self.root.withdraw()
            self.root.update()
            time.sleep(0.3)

            image_path = capture_full_screen()

            self.root.deiconify()
            if not auto:
                self.root.lift()
            self.root.update()

            self.last_capture_path = image_path
            self.capture_path_text.config(text=image_path)

            self.status_label.config(text="\u25cf  Running OCR...", fg=FG_DIM)
            self.root.update_idletasks()

            extracted_text = extract_text_from_image(image_path)

            self.status_label.config(text="\u25cf  Classifying...", fg=FG_DIM)
            self.root.update_idletasks()

            risk_score = classify_text(extracted_text, self.model, self.tfidf)
            risk_band = interpret_risk(risk_score)

            self.score_label.config(text=f"{risk_score:.3f}")
            self.band_label.config(text=risk_band)
            self.set_risk_color(risk_band)
            self._last_risk_band = risk_band

            self.ocr_textbox.delete("1.0", tk.END)
            self.ocr_textbox.insert(tk.END, extracted_text if extracted_text else "[No text detected]")

            self.status_label.config(text="\u25cf  Scan complete", fg=SUCCESS)

        except Exception as e:
            self.root.deiconify()  # ensure app comes back on error
            self.root.lift()
            self.status_label.config(text="\u25cf  Error", fg=DANGER)
            messagebox.showerror("Error", str(e))

    def auto_scan_loop(self):
        self.scan_screen(auto=True)
        if self._last_risk_band in ("MEDIUM", "HIGH"):
            self.root.deiconify()
            self.root.wm_attributes("-topmost", True)
            self.root.after(100, lambda: self.root.wm_attributes("-topmost", False))
            self.root.focus_force()
        self.auto_scan_job = self.root.after(3000, self.auto_scan_loop)

    def toggle_auto_scan(self):
        if self.auto_scan_job is None:
            self.auto_button.config(text="\u23f9  Stop Auto Scan")
            self.status_label.config(text="\u25cf  Auto scan enabled", fg=SUCCESS)
            self.auto_scan_loop()
        else:
            self.root.after_cancel(self.auto_scan_job)
            self.auto_scan_job = None
            self.auto_button.config(text="\u25b6  Start Auto Scan (3s)")
            self.status_label.config(text="\u25cf  Auto scan stopped", fg=FG_DIM)


def main():
    root = tk.Tk()
    app = PhishingScannerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()