import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import torch

from model import load_model
from data import inference_transform
from log import get_logger

class DogCatApp:
    CLASS_NAMES = ['Cat', 'Dog']

    def __init__(self, root:tk.Tk, checkpoint_path: str = 'best_model.pth'):
        self.root = root
        self.root.title('Dog Cat Classification')
        self.logger = get_logger(name='GUI')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = load_model(checkpoint_path, self.device)
        self.logger.info(f'Model loaded from {checkpoint_path} on {self.device}')

        self.image_path:str | None = None
        self.img_tk: ImageTk.PhotoImage | None = None

        self._build_ui()

    def _build_ui(self):
        tk.Button(self.root, text="Choose image", command= self.load_image).pack(pady=10)
        self.image_label = tk.Label(self.root)
        self.image_label.pack()
        tk.Button(self.root, text="Predict", command=self.predict_image).pack(pady=10)
        self.prediction_label = tk.Label(self.root,text="", font=("Arial", 16))
        self.prediction_label.pack(pady=10)

    def load_image(self):
        file_path= filedialog.askopenfilename(
            title='Choose image',
            filetypes=[("Image files", "*.png *.jpg *.jpeg")]
        )
        if file_path:
            self.image_path = file_path
            img = Image.open(file_path).resize((256, 256))
            self.img_tk = ImageTk.PhotoImage(img)
            self.image_label.config(image=self.img_tk)
            self.image_label.image = self.img_tk
            self.prediction_label.config(text="")
            self.logger.debug(f"Image loaded from {file_path}")

    def predict_image(self):
        if self.image_path is None:
            messagebox.showerror('Error', 'No image selected.')
            return

        try:
            img= Image.open(self.image_path).convert('RGB')
            img_tensor = inference_transform(img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = model_out = self.model(img_tensor)
                _, predicted_idx = torch.max(output, 1)

            prediction = self.CLASS_NAMES[predicted_idx.item()]
            self.prediction_label.config(text=f"Prediction: {prediction}")
            self.logger.info(f"Prediction: {prediction} - {self.image_path}")
        except Exception as e:
            self.logger.error(f"Prediction error: {e}")
            messagebox.showerror('Error', f"Prediction error: {e}")


if __name__ == '__main__':
    window = tk.Tk()
    app = DogCatApp(window)
    window.mainloop()