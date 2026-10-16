"""Train the chihuahua-vs-muffin CNN with gaussian noise augmentation.

Same model + data as 1016.py, but the TRAIN transform corrupts inputs with a
gaussian filter: either additive gaussian noise ("gaussian") or a gaussian blur
("blur"). The test set is first evaluated clean, then (for gaussian) also on a
noise-corrupted test set to gauge robustness.

Run with:  uv run python noise.py
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

MODES = ["gaussian", "blur"]
NOISE_STD = 0.1
BLUR_KERNEL_SIZE = 5
BLUR_SIGMA = 1.0

# Best (batch_size, lr) found by tune.py
BATCH_SIZE = 16
LR = 0.001

device = "cuda" if torch.cuda.is_available() else "cpu"

_NORMALIZE = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


class AddGaussianNoise:
    """Add zero-mean gaussian noise to a tensor in [0, 1] space."""

    def __init__(self, std=0.1):
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn_like(tensor) * self.std


def make_train_transform(mode):
    if mode == "gaussian":
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                AddGaussianNoise(NOISE_STD),
                _NORMALIZE,
            ]
        )
    if mode == "blur":
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.GaussianBlur(kernel_size=BLUR_KERNEL_SIZE, sigma=BLUR_SIGMA),
                transforms.ToTensor(),
                _NORMALIZE,
            ]
        )
    raise ValueError(f"unknown mode: {mode}")


_TEST_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        _NORMALIZE,
    ]
)


class Net(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 53 * 53, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

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


def run_mode(mode):
    print(f"\n===== mode={mode} =====")
    set_seed(SEED)

    trainset = torchvision.datasets.ImageFolder(
        "./archive/train", transform=make_train_transform(mode)
    )
    testset = torchvision.datasets.ImageFolder(
        "./archive/test", transform=_TEST_TRANSFORM
    )
    classes = trainset.classes

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

    net = Net(len(classes)).to(device)
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

    clean_acc = evaluate(net, testloader)
    print(f"Test accuracy (clean): {clean_acc:.1f} %")

    if mode == "gaussian":
        noisy_testset = torchvision.datasets.ImageFolder(
            "./archive/test", transform=make_train_transform("gaussian")
        )
        noisy_testloader = torch.utils.data.DataLoader(
            noisy_testset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
        )
        noisy_acc = evaluate(net, noisy_testloader)
        print(f"Test accuracy (noisy eval): {noisy_acc:.1f} %")


def main():
    print(f"device={device}  epochs={EPOCHS}  batch_size={BATCH_SIZE}  lr={LR}")
    for mode in MODES:
        run_mode(mode)


if __name__ == "__main__":
    main()
