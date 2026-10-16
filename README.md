# Chihuahua vs Muffin

## 개요

치와와 머핀 구분해보기

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
