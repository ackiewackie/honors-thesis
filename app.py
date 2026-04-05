import tkinter as tk
from tkinter import scrolledtext, messagebox
import time

from model_utils import load_model, classify_text, interpret_risk
from ocr_utils import extract_text_from_image
from screen_capture import capture_full_screen


class PhishingScannerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Real-Time Phishing Scanner")
        self.root.geometry("520x700")

        self.model, self.tfidf = load_model()
        self.auto_scan_job = None
        self.last_capture_path = None

        self.build_ui()

    def build_ui(self):
        title = tk.Label(
            self.root,
            text="Phishing Detection Side Panel",
            font=("Arial", 16, "bold")
        )
        title.pack(pady=10)

        self.scan_button = tk.Button(
            self.root,
            text="Scan Current Screen",
            command=self.scan_screen,
            width=25,
            height=2
        )
        self.scan_button.pack(pady=8)

        self.auto_button = tk.Button(
            self.root,
            text="Start Auto Scan (5s)",
            command=self.toggle_auto_scan,
            width=25,
            height=2
        )
        self.auto_button.pack(pady=8)

        self.status_label = tk.Label(
            self.root,
            text="Status: Ready",
            font=("Arial", 11)
        )
        self.status_label.pack(pady=6)

        self.score_label = tk.Label(
            self.root,
            text="Risk Score: -",
            font=("Arial", 13, "bold")
        )
        self.score_label.pack(pady=6)

        self.band_label = tk.Label(
            self.root,
            text="Risk Band: -",
            font=("Arial", 13, "bold")
        )
        self.band_label.pack(pady=6)

        capture_label = tk.Label(
            self.root,
            text="Last Capture Path:",
            font=("Arial", 11, "bold")
        )
        capture_label.pack(pady=(12, 4))

        self.capture_path_text = tk.Label(
            self.root,
            text="None",
            wraplength=470,
            justify="left",
            font=("Arial", 10)
        )
        self.capture_path_text.pack(pady=(0, 10))

        ocr_label = tk.Label(
            self.root,
            text="Extracted OCR Text:",
            font=("Arial", 11, "bold")
        )
        ocr_label.pack(pady=(10, 4))

        self.ocr_textbox = scrolledtext.ScrolledText(
            self.root,
            wrap=tk.WORD,
            width=60,
            height=20
        )
        self.ocr_textbox.pack(padx=10, pady=8)

    def set_risk_color(self, band):
        if band == "HIGH":
            self.band_label.config(fg="red")
        elif band == "MEDIUM":
            self.band_label.config(fg="orange")
        else:
            self.band_label.config(fg="green")

    def scan_screen(self):
        try:
            self.status_label.config(text="Status: Preparing capture...")
            self.root.update_idletasks()

            # Hide the app so it doesn't appear in screenshot
            self.root.withdraw()
            self.root.update()
            time.sleep(0.3)  # small delay to ensure it's gone

            image_path = capture_full_screen()

            # Bring app back
            self.root.deiconify()
            self.root.lift()
            self.root.update()

            self.last_capture_path = image_path
            self.capture_path_text.config(text=image_path)

            self.status_label.config(text="Status: Running OCR...")
            self.root.update_idletasks()

            extracted_text = extract_text_from_image(image_path)

            self.status_label.config(text="Status: Classifying...")
            self.root.update_idletasks()

            risk_score = classify_text(extracted_text, self.model, self.tfidf)
            risk_band = interpret_risk(risk_score)

            self.score_label.config(text=f"Risk Score: {risk_score:.3f}")
            self.band_label.config(text=f"Risk Band: {risk_band}")
            self.set_risk_color(risk_band)

            self.ocr_textbox.delete("1.0", tk.END)
            self.ocr_textbox.insert(tk.END, extracted_text if extracted_text else "[No text detected]")

            self.status_label.config(text="Status: Scan complete")

        except Exception as e:
            self.root.deiconify()  # ensure app comes back on error
            self.root.lift()
            self.status_label.config(text="Status: Error")
            messagebox.showerror("Error", str(e))

    def auto_scan_loop(self):
        self.scan_screen()
        self.auto_scan_job = self.root.after(5000, self.auto_scan_loop)

    def toggle_auto_scan(self):
        if self.auto_scan_job is None:
            self.auto_button.config(text="Stop Auto Scan")
            self.status_label.config(text="Status: Auto scan enabled")
            self.auto_scan_loop()
        else:
            self.root.after_cancel(self.auto_scan_job)
            self.auto_scan_job = None
            self.auto_button.config(text="Start Auto Scan (5s)")
            self.status_label.config(text="Status: Auto scan stopped")


def main():
    root = tk.Tk()
    app = PhishingScannerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()