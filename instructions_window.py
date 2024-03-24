import tkinter as tk
from PIL import ImageTk, Image


class InstructionsWindow(tk.Toplevel):
    def __init__(self):
        super().__init__()
        self.title("Instructions")
        self.geometry("400x540")

        # loading and resizing the images
        image_button = Image.open("instruction_photos/captureButton.png")
        image_button = image_button.resize((250, 55))
        image_frame = Image.open("instruction_photos/Frame.png")
        image_frame = image_frame.resize((200, 200))

        # converting the images to Tkinter compatible format
        image_button_tk = ImageTk.PhotoImage(image_button)
        image_frame_tk = ImageTk.PhotoImage(image_frame)

        # creating and packing the labels
        text_label1 = tk.Label(self, text="To receive a verification follow these instructions:", font=("Helvetica", 12))
        text_label1.pack()

        empty_label0 = tk.Label(self)
        empty_label0.pack()

        text_label2 = tk.Label(self, text="1. Position your face within the frame", font=("Helvetica", 12))
        text_label2.pack()

        image_frame_label = tk.Label(self, image=image_frame_tk)
        image_frame_label.pack()

        empty_label1 = tk.Label(self)
        empty_label1.pack()

        text_label3 = tk.Label(self, text="2. Click the Capture button to capture the image", font=("Helvetica", 12))
        text_label3.pack()

        image_button_label = tk.Label(self, image=image_button_tk)
        image_button_label.pack()

        empty_label2 = tk.Label(self)
        empty_label2.pack()

        text_label4 = tk.Label(self, text="3. Wait for verification", font=("Helvetica", 12))
        text_label4.pack()

        text_label5 = tk.Label(self, text="Verifying...\n\n", font=("Helvetica", 14))
        text_label5.pack()

        self.image_references = [image_frame_tk, image_button_tk]

        self.ok_button = tk.Button(self, text="OK", command=self.destroy)
        self.ok_button.pack()


if __name__ == '__main__':
    instructions_window = InstructionsWindow()
    instructions_window.mainloop()
