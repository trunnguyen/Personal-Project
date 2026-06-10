import torch
import torch.nn as nn

class SpeechLSTM(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=64, num_classes=4):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        """
        x expected shape: (B, T, F)
        But we defensively fix common mistakes:
        - (B, 1, T, F) → squeeze channel dim
        """

        # 🔒 ABSOLUTE SAFETY
        if x.dim() == 4:
            x = x.squeeze(1)  # (B, T, F)

        if x.dim() != 3:
            raise ValueError(f"LSTM expected 3D input (B,T,F), got {x.shape}")

        _, (h, _) = self.lstm(x)
        out = self.fc(h[-1])
        return out

