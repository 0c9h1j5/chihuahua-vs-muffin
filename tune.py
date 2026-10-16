"""Hyperparameter sweep for the chihuahua-vs-muffin CNN (see 1016.py).

Finds a good (batch_size, lr) for SGD + momentum on the ImageFolder dataset at
./archive/train and ./archive/test.

Run with:  uv run python tune.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

SEED = 42
EPOCHS = 6
NUM_WORKERS = 4
BATCH_SIZES = [16, 32, 64]
LRS = [0.001, 0.005, 0.01]

device = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

trainset = torchvision.datasets.ImageFolder(root="./archive/train", transform=transform)
testset = torchvision.datasets.ImageFolder(root="./archive/test", transform=transform)

classes = trainset.classes


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 53 * 53, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, len(classes))

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def build_loaders(batch_size):
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        persistent_workers=NUM_WORKERS > 0,
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        persistent_workers=NUM_WORKERS > 0,
    )
    return trainloader, testloader


def evaluate(net, loader):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


def train_eval(trainloader, testloader, lr):
    """Train a fresh Net for EPOCHS and return test accuracy (%)."""
    set_seed(SEED)

    net = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)

    for epoch in range(EPOCHS):
        net.train()
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(
            f"    [epoch {epoch + 1:>2}/{EPOCHS}] loss: "
            f"{running_loss / len(trainloader):.3f}"
        )

    return evaluate(net, testloader)


def main():
    print(f"device={device}  epochs={EPOCHS}  seed={SEED}  workers={NUM_WORKERS}")
    print(f"train={len(trainset)}  test={len(testset)}  classes={classes}")

    results = []
    for batch_size in BATCH_SIZES:
        trainloader, testloader = build_loaders(batch_size)
        for lr in LRS:
            print(f"\n>>> batch_size={batch_size}  lr={lr}")
            acc = train_eval(trainloader, testloader, lr)
            results.append((batch_size, lr, acc))
            print(f"RESULT batch_size={batch_size:<3} lr={lr:<6} acc={acc:.1f}%")

    print("\n==== summary (sorted by accuracy) ====")
    for batch_size, lr, acc in sorted(results, key=lambda r: r[2], reverse=True):
        print(f"batch_size={batch_size:<3} lr={lr:<6} acc={acc:.1f}%")

    best = max(results, key=lambda r: r[2])
    print(f"\nBEST batch_size={best[0]} lr={best[1]} acc={best[2]:.1f}%")


if __name__ == "__main__":
    main()
