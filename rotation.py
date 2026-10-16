"""Train the chihuahua-vs-muffin CNN with rotation augmentation.

Same model + data as 1016.py, but the TRAIN transform additionally applies a
random rotation. The test set is always evaluated clean (no augmentation).

Run with:  uv run python rotation.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

SEED = 42
EPOCHS = 10
NUM_WORKERS = 4
ROTATION_DEGREES = 15

# Best (batch_size, lr) found by tune.py
BATCH_SIZE = 16
LR = 0.001

device = "cuda" if torch.cuda.is_available() else "cpu"

# Train: augment with rotation. Test: plain resize/normalize only.
train_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomRotation(ROTATION_DEGREES),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)
test_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

trainset = torchvision.datasets.ImageFolder(
    "./archive/train", transform=train_transform
)
testset = torchvision.datasets.ImageFolder("./archive/test", transform=test_transform)

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


def main():
    set_seed(SEED)
    print(
        f"device={device}  epochs={EPOCHS}  batch_size={BATCH_SIZE}  lr={LR}  "
        f"rotation={ROTATION_DEGREES}deg"
    )
    print(f"train={len(trainset)}  test={len(testset)}  classes={classes}")

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        persistent_workers=NUM_WORKERS > 0,
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        persistent_workers=NUM_WORKERS > 0,
    )

    net = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)

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
            f"[epoch {epoch + 1:>2}/{EPOCHS}] loss: "
            f"{running_loss / len(trainloader):.3f}"
        )

    acc = evaluate(net, testloader)
    print(f"Test accuracy: {acc:.1f} %")


if __name__ == "__main__":
    main()
