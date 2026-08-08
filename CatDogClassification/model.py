import torch
import torch.nn as nn

#Setup CNN model

class DogCatCNN(nn.Module):
    def __init__(self, dropout: float = 0.5):
        super(DogCatCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3,stride=1, padding=1)
        self.bn1=nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3,stride=1, padding=1)
        self.bn2=nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size=2,stride= 2)

        #added hidden FC + Dropout before output
        self.fc1 = nn.Linear(32*32*32, 512)
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 32*32*32)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x= self.fc2(x)
        return x

class DogCatCNNv2(nn.Module):
    def __init__(self, dropout: float = 0.3):
        super().__init__()
        def block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
        self.features = nn.Sequential(
            block(3, 32), block(32, 64), block(64, 128), block(128, 256),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(256, 2)

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)


def load_model(checkpoint_path: str, device: torch.device, model_cls=DogCatCNN):
    #Load a saved Model
    model = model_cls()
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model