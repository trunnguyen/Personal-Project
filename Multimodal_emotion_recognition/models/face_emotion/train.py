import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from cnn import FaceCNN

X = np.load(r"C:\Users\nguye\OneDrive\Documents\Data_Science\Multimodal_emotion_recognition\Data\processed\fer2013\fer2013_faces.npy")
y = np.load(r"C:\Users\nguye\OneDrive\Documents\Data_Science\Multimodal_emotion_recognition\Data\processed\fer2013\fer2013_labels.npy")

X = torch.tensor(X).unsqueeze(1).float()
y = torch.tensor(y).long()

loader = DataLoader(TensorDataset(X, y), batch_size=64, shuffle=True)

model = FaceCNN()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(5):
    for xb, yb in loader:
        pred = model(xb)
        loss = criterion(pred, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "face_model.pth")
