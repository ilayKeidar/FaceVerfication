import tkinter as tk


class AboutWindow(tk.Toplevel):
    def __init__(self):
        super().__init__()
        self.title("About")

        self.text = (
            "This is the About Window. Here, I'll explain a little on my project and face verification in general. \n\n"
            "In this project, I created a face verification system using a siamese neural network,"
            " a machine learning model. \n\nWhen clicking on the Try the Face Verification button, you will be directed"
            " to a new window, there you can try out the face verification system and look at some of the statistics"
            " of your verification"
        )

        self.text_widget = tk.Text(self, width=50, height=10, font=("Helvetica", 12), wrap="word")
        self.text_widget.insert(tk.END, self.text)
        self.text_widget.pack(expand=True, fill=tk.BOTH)

        self.ok_button = tk.Button(self, text="OK", command=self.destroy)
        self.ok_button.pack()
