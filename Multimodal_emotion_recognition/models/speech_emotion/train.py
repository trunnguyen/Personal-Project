import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from lstm import SpeechLSTM

X = torch.tensor(np.load(r"C:\Users\nguye\OneDrive\Documents\Data_Science\Multimodal_emotion_recognition\Data\processed\ravdess\ravdess_audio.npy")).float()
y = torch.tensor(np.load(r"C:\Users\nguye\OneDrive\Documents\Data_Science\Multimodal_emotion_recognition\Data\processed\ravdess\ravdess_labels.npy")).long()

loader = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)

model = SpeechLSTM()
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

torch.save(model.state_dict(), "speech_model.pth")
