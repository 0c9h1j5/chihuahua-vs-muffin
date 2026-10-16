import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # [수정3] F.relu 쓰려면 필요
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [
        # 이미지는 augment.py에서 미리 224x224로 줄여 둠
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

batch_size = 32

trainset = torchvision.datasets.ImageFolder(
    root="./archive/train_224", transform=transform
)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True, num_workers=0
)

testset = torchvision.datasets.ImageFolder(
    root="./archive/test_224", transform=transform
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=False, num_workers=0
)


classes = trainset.classes  # [수정2] 폴더 이름 = 클래스
print(len(trainset), classes, len(testset))


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 53 * 53, 120)  # [수정5] 224 입력 기준 계산
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, len(classes))

    def forward(self, x):  # [수정4] 들여쓰기: __init__ 밖으로
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

net = Net().to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

start_time = time.time()


for epoch in range(10):  # [수정7] 데이터가 적어서 에폭↑, lr↑
    net.train()
    running_loss = 0.0
    for data in trainloader:
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"[epoch {epoch + 1}] loss: {running_loss / len(trainloader):.3f}")


finish_time = time.time()

print("Finished Training", finish_time - start_time)


PATH = "./yes_no_net.pth"
torch.save(net.state_dict(), PATH)

dataiter = iter(testloader)
images, labels = next(dataiter)
imshow(torchvision.utils.make_grid(images))
print("GroundTruth: ", " ".join(f"{classes[labels[j]]:5s}" for j in range(batch_size)))


# [수정10] 예측 + 정확도 (튜토리얼 뒷부분)
net.eval()
with torch.no_grad():
    outputs = net(images.to(device))
    _, predicted = torch.max(outputs, 1)
print(
    "Predicted:   ", " ".join(f"{classes[predicted[j]]:5s}" for j in range(batch_size))
)

correct = 0
total = 0
with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f"Test accuracy: {100 * correct / total:.1f} %  ({correct}/{total})")
