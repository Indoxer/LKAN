import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from sympy import Max
from tqdm import tqdm

from lkan.models import KANConv2d, KANLinear, KANLinearFFT

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 10),
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class KAN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            KANConv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.MaxPool2d(2),
            KANConv2d(16, 6, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            KANLinearFFT(6 * 7 * 7, 10),
        )

    def forward(self, x):
        x = self.layers(x)
        return x


save_dir = ".experiments/"
data_dir = ".data/"
batch_size = 64
split_ratio = 0.8

lr = 0.0003
epochs = 5

transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
    ]
)

dataset = torchvision.datasets.MNIST(data_dir, transform=transform, download=True)
ds_train, ds_val = torch.utils.data.random_split(
    dataset,
    [int(len(dataset) * split_ratio), len(dataset) - int(len(dataset) * split_ratio)],
)
loader_val = torch.utils.data.DataLoader(ds_val, batch_size=batch_size, shuffle=False)
loader_train = torch.utils.data.DataLoader(
    ds_train, batch_size=batch_size, shuffle=True
)

# model = MLP().to(device)
model = KAN().to(device)

counter = 0
for param in model.parameters():
    counter += param.numel()

print(f"Number of parameters: {counter}")

opt = torch.optim.Adam(model.parameters(), lr=lr)

# prof = torch.profiler.profile(
#     on_trace_ready=torch.profiler.tensorboard_trace_handler(".logs/conv"),
#     record_shapes=True,
#     profile_memory=True,
#     with_stack=True,
# )

# prof.start()

for epoch in range(epochs):
    train_acc = 0
    avg_train_loss = 0
    for x, y in tqdm(loader_train):
        x, y = x.to(device), y.to(device)

        opt.zero_grad()

        y_pred = model(x)
        loss = nn.CrossEntropyLoss()(y_pred, y)

        loss.backward()
        opt.step()
        # prof.step()

        train_acc += (y_pred.argmax(1) == y).float().mean().item()
        avg_train_loss += loss.item()

    avg_train_loss /= len(loader_train)
    train_acc /= len(loader_train)

    with torch.no_grad():
        acc = 0
        avg_loss = 0
        for x, y in loader_val:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = nn.CrossEntropyLoss()(y_pred, y)
            acc += (y_pred.argmax(1) == y).float().mean().item()
            avg_loss += loss.item()
            # prof.step()
        acc /= len(loader_val)
        avg_loss /= len(loader_val)
        print(
            f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss} - Val Loss: {avg_loss} - Train Acc: {train_acc} -Val Acc: {acc}"
        )

# prof.stop()
