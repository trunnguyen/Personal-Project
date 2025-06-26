import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk
import torch
import torch.nn as nn
import torchvision.transforms as transforms


class DogCatCNN(nn.Module):
    def __init__(self):
        super(DogCatCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 32 * 32, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 32 * 32)
        x = self.fc1(x)
        return x


model = DogCatCNN()
model.load_state_dict(torch.load(
    'best_model.pth', map_location=torch.device('cpu')))
model.eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def load_image():
    global image_path, img_tk
    file_path = filedialog.askopenfilename(
        title="Chọn ảnh chó hoặc mèo",
        filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )
    if file_path:
        image_path = file_path
        img = Image.open(image_path)
        img = img.resize((256, 256))
        img_tk = ImageTk.PhotoImage(img)
        image_label.config(image=img_tk)
        image_label.image = img_tk
        prediction_label.config(text="")


def predict_image():
    if 'image_path' not in globals():
        messagebox.showerror("Lỗi", "Vui lòng chọn một ảnh trước.")
        return

    try:
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0)

        with torch.no_grad():
            output = model(img_tensor)
            _, predicted_idx = torch.max(output, 1)
            prediction = "Mèo" if predicted_idx.item() == 1 else "Chó"
            prediction_label.config(text=f"Dự đoán: {prediction}")

    except Exception as e:
        messagebox.showerror(
            "Lỗi dự đoán", f"Đã xảy ra lỗi trong quá trình dự đoán: {e}")


window = tk.Tk()
window.title("Ứng dụng dự đoán Chó/Mèo")

load_button = tk.Button(window, text="Chọn ảnh", command=load_image)
load_button.pack(pady=10)

image_label = tk.Label(window)
image_label.pack()

predict_button = tk.Button(window, text="Dự đoán", command=predict_image)
predict_button.pack(pady=10)

prediction_label = tk.Label(window, text="", font=("Arial", 16))
prediction_label.pack(pady=10)

window.mainloop()
