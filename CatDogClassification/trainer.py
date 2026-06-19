import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np

from log import get_logger

#Model training
def train(model, train_loader, val_loader, device,
          epochs: int = 20, patience: int =5,
          checkpoint_path: str = 'best_model.pth',
          class_names: list = None,):

    logger = get_logger()
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=0.001)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                  patience=3)

    history= {
        'train_loss': [], 'train_acc': [],
        'val_loss':   [], 'val_acc':   [],
    }

    best_val_loss = float('inf')
    best_val_acc = 0.0
    epochs_no_improvement = 0
    logger.info(f'Starting training - device: {device}, epochs: {epochs}, patience: {patience}')


    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
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
        epoch_acc = 100 * correct / total
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)

        val_loss, val_acc = evaluate(model, val_loader, device, criterion)
        history['val_loss'].append(epoch_loss)
        history['val_acc'].append(epoch_acc)

        current_lr = optimizer.param_groups[0]['lr']
        logger.info(
            f"Epoch {epoch+1:>3} | "
            f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}% | "
            f"Val loss: {val_loss:.4f} Acc: {val_acc:.2f}% | "
            f"LR: {current_lr:.6f}"
        )

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = epoch_acc
            epochs_no_improvement = 0
            torch.save(model.state_dict(), checkpoint_path)
            logger.debug(f" Best model saved (val_loss= {val_loss:.4f}), val_acc= {val_acc:.2f}%)")
        else:
            epochs_no_improvement += 1
            if epochs_no_improvement == patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs!")
                break

    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    logger.info(f"Training complete - best val accuracy: {best_val_acc:.2f}%")
    return history

def evaluate(model, val_loader, device, criterion=None):
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(val_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

def plot_history(history: dict):
    epochs = range(1,len(history['train_loss']) + 1)

    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.plot(epochs, history['train_loss'], marker='o',label='Training')
    plt.plot(epochs, history['val_loss'], marker='o',label='Validation loss')
    plt.title("Training and validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(epochs, history['train_acc'], marker='o',label='Training Accuracy')
    plt.plot(epochs, history['val_acc'], marker='o',label='Validation Accuracy')
    plt.title("Training and validation accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

def visualize_predictions(model, val_loader, device,
                          class_names: list = None, num_images: int = 5):
    if class_names is None:
        class_names = getattr(val_loader.dataset, 'class_names', ['Class 0', 'Class 1'])

    model.eval()
    images, labels = next(iter(val_loader))
    images, labels = images.to(device), labels.to(device)

    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

    num_images = min(num_images, images.size(0))
    fig, axes = plt.subplots(1, num_images, figsize=(15,3))
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.stack([0.229, 0.224, 0.225])

    for i in range(num_images):
        img = images[i].cpu().numpy().transpose(1, 2, 0)
        img = std * img + mean
        img = np.clip(img, 0, 1)
        axes[i].imshow(img)
        pred_name = class_names[predicted[i].item()]
        true_name = class_names[labels[i].item()]
        color = 'green' if predicted[i] == labels[i] else 'red'
        axes[i].set_title(f"Pred: {pred_name}\nTrue: {true_name}", color = color)
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()