# Chihuahua vs Muffin

## 개요

치와와 머핀 구분해보기
DATASET : https://www.kaggle.com/datasets/samuelcortinhas/muffin-vs-chihuahua-image-classification?resource=download-directory

## 1차 시도

batch_size = 4
lr=0.005

sample: 4733 ['chihuahua', 'muffin'] test : 1184

Test accuracy: 82.2 %  (973/1184)

## 2차 시도

batch_size = 16
lr=0.001

sample: 4733 ['chihuahua', 'muffin'] test : 1184

Test accuracy: 85.6 %  (1013/1184)

## 3차 시도

batch_size = 32
lr=0.001

sample: 18932 ['chihuahua', 'muffin'] test : 1184

Finished Training 458.16535568237305

Test accuracy: 86.4 %  (1023/1184)

### 특이사항

- 이미지의 갯수를 증가(샤프, 노이즈, 회전 등)
- 이미지를 미리 224 * 224로 만들어 시간을 단축시켰다.

## 4차 시도

batch_size = 32
lr=0.001

sample: train 16092 / val 710 / test 1184 ['chihuahua', 'muffin']

Finished Training 323.5018062591553

Best val accuracy: 86.3 %

Test accuracy: 85.9 %  (1017/1184)

### 특이사항

- train 원본의 15%를 검증셋(val_224)으로 분리 (원본 단위로 나눠 증강본이 섞이지 않게 함, 검증셋은 증강 안 함)
- 매 에폭 검증 정확도를 보고, 가장 좋은 모델만 저장해서 test는 마지막에 한 번만 평가
- 6에폭 이후 학습 loss는 계속 떨어지는데(0.100 → 0.017) 검증 정확도는 86% 근처에서 멈춤 → 과적합

## 5차 시도

batch_size = 32
lr=0.001

sample: train 16092 / val 710 / test 1184 ['chihuahua', 'muffin']

Finished Training 336.80426239967346

Best val accuracy: 88.2 %

Test accuracy: 87.4 %  (1035/1184)

### 특이사항

- conv 층을 하나 추가: conv3 = Conv2d(16, 32, 3), fc1 입력 32 * 25 * 25
- 6에폭에서 검증 88.2%로 최고, 이후 학습 loss는 계속 떨어지는데(0.218 → 0.051) 검증 정확도는 84~87%로 내려감 → 여전히 과적합

## 6차 시도

batch_size = 32
lr=0.001
epoch = 40

sample: train 4023 / val 710 / test 1184 ['chihuahua', 'muffin']

Finished Training 734.043553352356

Best val accuracy: 89.6 %

Test accuracy: 90.8 %  (1075/1184)

### 특이사항

- 미리 만든 복사본(회전/샤프/노이즈) 대신 온라인 증강으로 변경: 사진을 불러올 때마다 랜덤 변형
- 좌우 반전, ±25도 회전, 10% 흑백, 50% 샤프닝, 가우시안 노이즈 (학습셋에만 적용, val/test는 그대로)
- 학습 폴더는 원본만 224로 줄인 train_224_orig (4·5차와 같은 train/val 나눔)
- 한 에폭이 4,023장이라 같은 양을 보도록 10 → 40에폭
- 증강 CPU 비용은 사진당 약 3.5ms로 크지 않음
- 과적합이 사라짐: 학습 loss가 0.23에서 멈춰 있고 검증 정확도는 마지막 에폭(89.6%)까지 계속 오름 → 에폭을 더 늘리면 더 오를 여지가 있음

## 7차 시도

batch_size = 32
lr=0.001 (CosineAnnealingLR로 0까지 감소)
epoch = 80

sample: train 4023 / val 710 / test 1184 ['chihuahua', 'muffin']

Finished Training 2081.4734132289886

Best val accuracy: 92.7 % (65에폭)

Test accuracy: 92.8 %  (1099/1184)

### 특이사항

- 모델 구조 변경 (파라미터 약 240만 → 42만 개)
  - conv 블록 4개 (3x3 conv → BatchNorm → ReLU → MaxPool), 채널 32 → 64 → 128 → 256
  - flatten 대신 AdaptiveAvgPool2d(1), 분류층에 Dropout(0.3)
- 40 → 80에폭, CosineAnnealingLR로 학습률을 점점 줄임
- 증강/데이터는 6차와 동일
- 초반 10에폭은 검증 정확도가 60%대로 느리게 오르다가 20에폭 이후 빠르게 상승
- 학습 시간이 약 35분으로 6차(12분)의 3배 가까이 걸림 (모델이 커지고 에폭이 2배)

## 8차 시도

batch_size = 32
lr=0.001 (CosineAnnealingLR로 0까지 감소)
epoch = 40

sample: train 4023 / val 710 / test 1184 ['chihuahua', 'muffin']

Finished Training 807.8887932300568

Best val accuracy: 94.1 % (21에폭)

Test accuracy: 93.8 %  (1110/1184)

### 특이사항

- 블록마다 conv를 2개로 (conv → BN → ReLU) x 2 → pool, VGG 방식 (파라미터 42만 → 121만 개)
- 80에폭은 너무 오래 걸려서 40에폭으로 줄임
- DataLoader num_workers=4 (+ persistent_workers, pin_memory)로 에폭당 약 26초 → 20초
  - Windows에서 worker를 쓰려고 학습 코드를 main() 안으로 옮기고 `if __name__ == "__main__":`로 감쌈
  - worker를 6개로 늘려도 같은 속도 → 이제 GPU 계산이 병목
- 검증 정확도가 에폭마다 83~94%로 크게 흔들림 (검증셋이 710장이라 몇 장 차이로도 변동이 큼)

## 9차 시도

batch_size = 32
lr=0.001 (CosineAnnealingLR로 0까지 감소)
epoch = 40

sample: train 4023 / val 710 / test 1184 ['chihuahua', 'muffin']

Finished Training 537.7116229534149

Best val accuracy: 94.6 % (38에폭)

Test accuracy: 94.6 %  (1120/1184)

### 특이사항

- 혼합 정밀도(AMP) 적용: torch.autocast(float16) + GradScaler, 모델/설정은 8차와 동일
- 배치당 145ms → 97ms, GPU 메모리 1.9GB → 1.1GB
- 학습 시간 808초 → 538초 (약 1.5배 빨라짐)
- 정확도는 떨어지지 않음 (테스트 93.8% → 94.6%, 랜덤 요소가 있어서 이 정도 차이는 실행마다 달라질 수 있음)
