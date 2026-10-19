import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import v2

transform = transforms.Compose(
    [
        # 이미지는 archive/*_224 폴더에 미리 224x224로 줄여 둠
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

# 학습용: 사진을 불러올 때마다 랜덤으로 변형 (온라인 증강). val/test는 위 transform
train_transform = v2.Compose(
    [
        v2.ToImage(),
        v2.RandomHorizontalFlip(),  # 좌우 반전
        v2.RandomRotation(25),  # ±25도 회전
        v2.RandomGrayscale(p=0.1),  # 10% 확률로 흑백
        v2.RandomAdjustSharpness(2, p=0.5),  # 50% 확률로 샤프닝
        v2.ToDtype(torch.float32, scale=True),
        v2.GaussianNoise(sigma=0.05),  # 노이즈 (0~1 텐서로 바꾼 뒤에 적용)
        v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

batch_size = 32

# 사진 불러오기 + 증강을 CPU 여러 개로 나눠서 함 (GPU가 계산하는 동안 다음 배치 준비)
num_workers = 4


def make_loader(root, tf, shuffle, workers):
    dataset = torchvision.datasets.ImageFolder(root=root, transform=tf)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        persistent_workers=True,  # 에폭마다 worker를 새로 띄우지 않음
        pin_memory=True,  # GPU로 옮기는 속도 향상
    )
    return dataset, loader


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def conv_block(in_ch, out_ch):
    """(3x3 conv -> BatchNorm -> ReLU) x 2 -> 2x2 pool (크기 절반)"""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),  # 층마다 값 분포를 맞춰 학습을 빠르고 안정적으로
        nn.ReLU(),
        # 같은 크기에서 conv를 한 번 더: 줄이기 전에 특징을 한 번 더 뽑음 (VGG 방식)
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
    )


class Net(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # 채널을 늘림: 3 -> 32 -> 64 -> 128 -> 256, 크기 224 -> 112 -> 56 -> 28 -> 14
        self.features = nn.Sequential(
            conv_block(3, 32),
            conv_block(32, 64),
            conv_block(64, 128),
            conv_block(128, 256),
        )
        # flatten 대신 채널별 평균 (256x14x14 -> 256) : fc 파라미터가 크게 줄어듦
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),  # 학습 때 일부 뉴런을 랜덤으로 꺼서 과적합 방지
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(self.gap(x), 1)
        return self.classifier(x)


def autocast(device):
    """혼합 정밀도(AMP): GPU에서 무거운 계산(conv, 행렬곱)을 float16으로.
    loss, BatchNorm 등 정밀도가 중요한 계산은 PyTorch가 알아서 float32로 둔다."""
    return torch.autocast(
        device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
    )


def evaluate(net, loader, device):
    """loader 전체에 대한 (맞춘 개수, 전체 개수)를 돌려준다."""
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            with autocast(device):
                outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct, total


def main():
    # 원본만 224로 줄인 폴더 (복사본 없음, val_224와 같은 나눔)
    trainset, trainloader = make_loader(
        "./archive/train_224_orig", train_transform, True, num_workers
    )
    # 검증셋: 에폭별 성능 확인 + 가장 좋은 모델 고르기용 (test는 마지막에 한 번만)
    valset, valloader = make_loader("./archive/val_224", transform, False, 2)
    testset, testloader = make_loader("./archive/test_224", transform, False, 2)

    classes = trainset.classes  # [수정2] 폴더 이름 = 클래스
    print(f"train {len(trainset)} / val {len(valset)} / test {len(testset)}", classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    net = Net(len(classes)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # 9차(40에폭)와 비교하려고 80에폭으로 늘림 (스케줄러도 80에폭에 맞춰짐)
    num_epochs = 80
    # 학습률을 코사인 곡선으로 0.001 -> 0 까지 천천히 줄임 (마지막에 안정적으로 수렴)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # float16은 아주 작은 기울기가 0이 되기 쉬워서, loss를 키워서 계산한 뒤 되돌림
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    PATH = "./yes_no_net.pth"
    best_val_acc = 0.0
    start_time = time.time()

    for epoch in range(num_epochs):
        net.train()
        running_loss = 0.0
        for data in trainloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            with autocast(device):
                outputs = net(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
        scheduler.step()
        val_correct, val_total = evaluate(net, valloader, device)
        val_acc = 100 * val_correct / val_total
        msg = f"[epoch {epoch + 1}] loss: {running_loss / len(trainloader):.3f}"
        print(f"{msg}  val acc: {val_acc:.1f} %  ({time.time() - start_time:.0f}s)")
        if val_acc > best_val_acc:  # 검증 정확도가 가장 좋은 모델만 저장
            best_val_acc = val_acc
            torch.save(net.state_dict(), PATH)

    finish_time = time.time()

    print("Finished Training", finish_time - start_time)
    print(f"Best val accuracy: {best_val_acc:.1f} %")

    net.load_state_dict(torch.load(PATH))  # 가장 좋았던 모델로 테스트

    dataiter = iter(testloader)
    images, labels = next(dataiter)
    imshow(torchvision.utils.make_grid(images))
    print(
        "GroundTruth: ", " ".join(f"{classes[labels[j]]:5s}" for j in range(batch_size))
    )

    # [수정10] 예측 + 정확도 (튜토리얼 뒷부분)
    net.eval()
    with torch.no_grad(), autocast(device):
        outputs = net(images.to(device))
        _, predicted = torch.max(outputs, 1)
    print(
        "Predicted:   ",
        " ".join(f"{classes[predicted[j]]:5s}" for j in range(batch_size)),
    )

    correct, total = evaluate(net, testloader, device)
    print(f"Test accuracy: {100 * correct / total:.1f} %  ({correct}/{total})")


# Windows에서 num_workers를 쓰면 worker들이 이 파일을 다시 import 하므로,
# 학습 코드는 직접 실행할 때만 돌도록 막아 둠
if __name__ == "__main__":
    main()
