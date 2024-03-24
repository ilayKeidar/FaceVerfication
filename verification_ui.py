import cv2
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import ttk
import os
import threading
import time
import numpy as np
from keras.models import load_model
from keras.utils import custom_object_scope

from testingModel import TestModel
from faceDataset import FaceData
from instructions_window import InstructionsWindow
from distanceLayer import L1Distance


class VerificationWindow(tk.Toplevel):

    def __init__(self, t_model):
        super().__init__()
        self.title("Face Verification")
        self.model = t_model

        self.cap = cv2.VideoCapture(0)

        self.webcam_label = tk.Label(self)
        self.webcam_label.grid(row=0, column=0, columnspan=2, padx=10, pady=10)

        separator = ttk.Separator(self, orient="horizontal")
        separator.grid(row=2, column=0, columnspan=2, sticky="ew", padx=10, pady=5)

        self.button_style1 = {"font": ("Helvetica", 12, "bold"), "bg": "#10a37f", "fg": "#ffffff", "bd": 0,
                         "relief": "raised", "width": 10}
        self.button_style2 = {"font": ("Helvetica", 12, "bold"), "bg": "#d9d9de", "fg": "#5b5c68", "bd": 0,
                         "relief": "raised", "width": 10}

        self.verification_label = tk.Label(self, text="Press Capture for verification",
                                           font=("Helvetica", 12), fg="#5b5c68")
        self.verification_label.grid(row=3, column=0, columnspan=2, padx=10, pady=5)

        self.capture_button = tk.Button(self, text="Capture", command=self.capture_image, **self.button_style1)
        self.capture_button.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")

        # Create the info button
        info_button = tk.Button(self, text="i", command=self.show_instructions,
                                font=("Helvetica", 10, "bold"), width=2, height=1)
        info_button.place(x=20, y=20)

        self.progress_bar = ttk.Progressbar(self, mode="determinate", length=200)
        self.progress_bar.grid(row=5, column=0, columnspan=2, padx=10, pady=5)

        self.capture_button.bind("<Enter>", lambda event: self.hover_button(self.capture_button, "#00A67E",
                                                                                "#008D68", hover=True))
        self.capture_button.bind("<Leave>", lambda event: self.hover_button(self.capture_button, "#00A67E",
                                                                                "#008D68", hover=False))

        self.update_webcam()

    def show_instructions(self):
        instructions_window = InstructionsWindow()

    def hover_button(self, button, default_color, changed_color, hover=False):
        if hover:
            button.config(bg=changed_color)
        else:
            button.config(bg=default_color)

    def update_webcam(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            frame = frame[100:100 + FaceData().capture_size, 300:300 + FaceData().capture_size, :]
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            image = Image.fromarray(frame)
            image = ImageTk.PhotoImage(image)
            self.webcam_label.configure(image=image)
            self.webcam_label.image = image

        self.webcam_label.after(10, self.update_webcam)

    def capture_image(self):
        ret, frame = self.cap.read()
        frame = cv2.flip(frame, 1)
        frame = frame[100:100 + FaceData().capture_size, 300:300 + FaceData().capture_size, :]
        cv2.imwrite(os.path.join('app_data', 'input', 'input_img.jpg'), frame)

        self.verification_label.grid_remove()

        verification_thread = threading.Thread(target=self.verify_image)
        verification_thread.start()

    def show_statistics(self, detection, results, verification):
        # opens a new window to show statistics
        statistics_window = tk.Toplevel(self)
        statistics_window.title("Verification Statistics")
        statistics_label_detection = tk.Label(statistics_window, text="Verification Score: " + str(verification*100) +
                                                                      "%\n\n", font=("Helvetica", 12))
        statistics_label_detection.pack()

        statistics_label_results = tk.Label(statistics_window, text="similarity to each photo based on the model (0 to"
                                                                    " 1, 1 being the most similar): \n\n "
                                                                    + str(results), font=("Helvetica", 12))
        statistics_label_results.pack()

    def update_progress(self, progress):
        self.progress_bar["value"] = progress

    def verify_image(self):
        def animate_verification_label():
            dots = ""
            while not self.verification_label_hidden:
                dots += "."
                if len(dots) > 3:
                    dots = ""
                self.verification_label.config(text="Verifying" + dots, fg="#5b5c68")
                time.sleep(0.5)

        def progress_bar_animation():
            max_progress = 28
            for progress in range(max_progress + 1):
                self.update_progress(progress * (100 / max_progress))
                time.sleep(1)

        self.verification_label_hidden = False
        self.verification_label.grid()

        animation_thread = threading.Thread(target=animate_verification_label)
        animation_thread.start()

        progress_thread = threading.Thread(target=progress_bar_animation)
        progress_thread.start()

        captured_image = cv2.imread(os.path.join('app_data', 'input', 'input_img.jpg'))

        results, verified, detection, verification = TestModel(self.model).verify(self.model, 0.32, 0.49)

        self.verification_label_hidden = True
        animation_thread.join()
        progress_thread.join()

        if verified:
            self.verification_label.config(text="Verified!", fg="green")
            print(detection)
        else:
            self.verification_label.config(text="Verification failed!", fg="red")
            print(detection)
        results2 = []
        for element in results:
            number = element[0][0]
            results2.append(number)

        # Create a button to show the statistics
        statistics_button = tk.Button(self, text="Statistics", command=lambda: self.show_statistics(detection, results2,
                                                                                                    verification),
                                      **self.button_style2)
        statistics_button.grid(row=4, column=0, columnspan=2, padx=10, pady=5)

        statistics_button.bind("<Enter>", lambda event: self.hover_button(statistics_button, "#d9d9de", "#b1b1b5",
                                                                          hover=True))
        statistics_button.bind("<Leave>", lambda event: self.hover_button(statistics_button, "#d9d9de", "#b1b1b5",
                                                                          hover=False))

    def __del__(self):
        self.cap.release()
        cv2.destroyAllWindows()


def run_verification_ui():
    with custom_object_scope({'L1Distance': L1Distance}):
        siamese_model = load_model('trained_model3.h5')

    face_ui = VerificationWindow(siamese_model)
    face_ui.mainloop()


if __name__ == '__main__':
    run_verification_ui()

