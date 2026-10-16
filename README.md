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
