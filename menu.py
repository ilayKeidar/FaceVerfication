import tkinter as tk
from tkinter import ttk
from keras.models import load_model
from keras.utils import custom_object_scope

from about import AboutWindow
from verification_ui import VerificationWindow
from distanceLayer import L1Distance


class MenuWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Face Verification Machine Learning Project")

        # window size
        self.geometry("750x350")

        # Create the main title label
        title_label = tk.Label(self, text="Face Verification Machine Learning Project",
                               font=("Helvetica", 16, "bold"))
        title_label.pack(pady=20)

        # line separator under the title
        separator = ttk.Separator(self, orient="horizontal")
        separator.pack(fill="x", pady=10)

        # Create the "About" button
        about_button = tk.Button(self, text="About", command=self.show_about, width=10, font=("Helvetica", 12))
        about_button.place(x=10, y=10)

        self.button_style = {"font": ("Helvetica", 14, "bold"), "bg": "#10a37f", "fg": "#ffffff", "bd": 0,
                             "relief": "raised", "width": 30}

        # Create the "Try the Face Verification" button
        verification_button = tk.Button(self, text="Try the Face Verification",
                                        command=self.open_verification_window, **self.button_style)

        verification_button.pack(pady=50)

        # using the hover_button function to create a darker button when the mouse is hovering over it
        verification_button.bind("<Enter>", lambda event: self.hover_button(verification_button, "#00A67E",
                                                                            "#008D68", hover=True))
        verification_button.bind("<Leave>", lambda event: self.hover_button(verification_button, "#00A67E",
                                                                            "#008D68", hover=False))

    def show_about(self):
        about_window = AboutWindow()

    def hover_button(self, button, default_color, changed_color, hover=False):
        if hover:
            button.config(bg=changed_color)
        else:
            button.config(bg=default_color)

    # loading the trained model and calling the VerificationWindow class
    def open_verification_window(self):
        with custom_object_scope({'L1Distance': L1Distance}):
            siamese_model = load_model('trained_model3.h5')

        verification_window = VerificationWindow(siamese_model)
        verification_window.mainloop()


if __name__ == "__main__":
    menu = MenuWindow()
    menu.mainloop()
