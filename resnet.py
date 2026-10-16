"""사전학습 ResNet18로 치와와 vs 머핀 분류 (전이학습).

ImageNet(120만 장)으로 미리 학습된 ResNet18을 가져와 마지막 층만 2클래스로 바꾸고,
전체를 작은 학습률로 조금 더 학습시킨다(fine-tuning).
데이터/증강/검증 방식은 1016.py와 같고, 정규화 값만 ImageNet 기준으로 바꾼다.

사용법:
    uv run --no-sync python resnet.py
"""

import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms import v2

# 사전학습 모델은 ImageNet 사진의 평균/표준편차로 정규화된 입력을 기대함
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# val/test용: 변형 없이 텐서 변환 + 정규화 (이미지는 augment.py에서 미리 224x224)
transform = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)

# 학습용: 1016.py와 같은 온라인 증강
train_transform = v2.Compose(
    [
        v2.ToImage(),
        v2.RandomHorizontalFlip(),  # 좌우 반전
        v2.RandomRotation(25),  # ±25도 회전
        v2.RandomGrayscale(p=0.1),  # 10% 확률로 흑백
        v2.RandomAdjustSharpness(2, p=0.5),  # 50% 확률로 샤프닝
        v2.ToDtype(torch.float32, scale=True),
        v2.GaussianNoise(sigma=0.05),  # 노이즈 (0~1 텐서로 바꾼 뒤에 적용)
        v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)

batch_size = 32
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


def build_model(num_classes):
    """ImageNet 사전학습 ResNet18의 마지막 층(1000클래스)을 num_classes로 교체."""
    net = resnet18(weights=ResNet18_Weights.DEFAULT)  # 처음 한 번 가중치 다운로드
    net.fc = nn.Linear(net.fc.in_features, num_classes)  # 512 -> 2, 새로 학습됨
    return net


def autocast(device):
    """혼합 정밀도(AMP): GPU에서 무거운 계산을 float16으로."""
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
    trainset, trainloader = make_loader(
        "./archive/train_224_orig", train_transform, True, num_workers
    )
    valset, valloader = make_loader("./archive/val_224", transform, False, 2)
    testset, testloader = make_loader("./archive/test_224", transform, False, 2)

    classes = trainset.classes
    print(f"train {len(trainset)} / val {len(valset)} / test {len(testset)}", classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    net = build_model(len(classes)).to(device)

    criterion = nn.CrossEntropyLoss()
    # 이미 잘 학습된 가중치를 망가뜨리지 않도록 작은 학습률로 전체를 조금씩 조정
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)

    # 사전학습 모델은 금방 수렴해서 에폭을 적게
    num_epochs = 10
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    PATH = "./resnet18_net.pth"
    best_val_acc = 0.0
    start_time = time.time()

    for epoch in range(num_epochs):
        net.train()
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
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

    print("Finished Training", time.time() - start_time)
    print(f"Best val accuracy: {best_val_acc:.1f} %")

    net.load_state_dict(torch.load(PATH))  # 가장 좋았던 모델로 테스트
    correct, total = evaluate(net, testloader, device)
    print(f"Test accuracy: {100 * correct / total:.1f} %  ({correct}/{total})")


# Windows에서 num_workers를 쓰면 worker들이 이 파일을 다시 import 하므로 막아 둠
if __name__ == "__main__":
    main()
