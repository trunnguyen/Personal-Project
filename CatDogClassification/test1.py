import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets

# 1. Tải dữ liệu (Data)
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets
train_dataset = datasets.ImageFolder(
    root='./data/train', transform=train_transform)
val_dataset = datasets.ImageFolder(
    root='./data/val', transform=val_transform)

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# 2. Định nghĩa mô hình CNN (Model)
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

# 3. Khởi tạo mô hình
model = DogCatCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001,
                    momentum=0.9)

device = torch.device("cuda" if torch.cuda.is_available()
                    else "cpu")
model.to(device)

# 4. Huấn luyện mô hình 
epochs = 20
train_loss_values = []
train_accuracy_values = []
val_loss_values = []
val_accuracy_values = []

# Early stopping 
patience = 5  
best_val_loss = float('inf')
epochs_no_improve = 0

for epoch in range(epochs):
    running_loss = 0.0
    correct = 0
    total = 0
    model.train()  
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = (100 * correct / total)
    train_loss_values.append(epoch_loss)
    train_accuracy_values.append(epoch_acc)
    print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

    model.eval()  
    val_running_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_epoch_loss = val_running_loss / len(val_loader)
    val_epoch_acc = (100 * val_correct / val_total)
    val_loss_values.append(val_epoch_loss)
    val_accuracy_values.append(val_epoch_acc)
    print(f"Epoch {epoch+1}, Val Loss: {val_epoch_loss:.4f}, Val Accuracy: {val_epoch_acc:.2f}%")

    if val_epoch_loss < best_val_loss:
        best_val_loss = val_epoch_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), 'best_model.pth') 
    else:
        epochs_no_improve += 1
        if epochs_no_improve == patience:
            print(f"Early stopping triggered after {epoch+1} epochs!")
            break

model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# Vẽ biểu đồ loss
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, len(train_loss_values) + 1), train_loss_values, marker='o', label='Train Loss')
plt.plot(range(1, len(val_loss_values) + 1), val_loss_values,
         marker='o', label='Validation Loss')
plt.title("Biểu đồ Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()

# Vẽ biểu đồ accuracy
plt.subplot(1, 2, 2)
plt.plot(range(1, len(train_accuracy_values) + 1), train_accuracy_values,
         marker='o', label='Train Accuracy')
plt.plot(range(1, len(val_accuracy_values) + 1), val_accuracy_values,
         marker='o', label='Validation Accuracy')
plt.title("Biểu đồ Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# 5. Đánh giá mô hình trên tập validation (với mô hình tốt nhất)
val_running_loss = 0.0
val_correct = 0
val_total = 0
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        val_running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        val_total += labels.size(0)
        val_correct += (predicted == labels).sum().item()

final_val_loss = val_running_loss / len(val_loader)
final_val_accuracy = (100 * val_correct / val_total)
print(
    f"\nĐộ chính xác cuối cùng trên tập validation (mô hình tốt nhất): {final_val_accuracy:.2f}%")
print(
    f"Loss cuối cùng trên tập validation (mô hình tốt nhất): {final_val_loss:.4f}")

# 6. Trực quan kết quả dự đoán
def visualize_prediction():
    model.eval()
    images, labels = next(iter(val_loader))
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

    num_images = min(5, images.size(0))
    fig, axes = plt.subplots(1, num_images, figsize=(15, 3))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    for i in range(num_images):
        img = images[i].cpu().numpy().transpose((1, 2, 0))
        img = std * img + mean
        img = np.clip(img, 0, 1)
        axes[i].imshow(img)
        axes[i].set_title(
            f"Pred: {'Chó' if predicted[i] == 0 else 'Mèo'}")
        axes[i].axis('off')
    plt.show()

visualize_prediction()
