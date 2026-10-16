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

## 3차 시도 (데이터 증강 + GPU)

batch_size = 32
lr=0.001

sample: 18932 ['chihuahua', 'muffin'] test : 1184

Finished Training 458.16535568237305

Test accuracy: 86.4 %  (1023/1184)

### 특이사항

- 이미지의 갯수를 증가(샤프, 노이즈, 회전 등)
- 이미지를 미리 224 * 224로 만들어 시간을 단축시켰다.

## 3차 변형 (bs=2, lr=0.002)

batch_size = 2
lr=0.002

sample: 4733 ['chihuahua', 'muffin'] test : 1184

Test accuracy: 86.1 %  (1019/1184)

## 하이퍼파라미터 튜닝 (tune.py)

batch_size = 16
lr=0.001

epochs = 6 sweep 결과 최고 성능:

sample: 4733 ['chihuahua', 'muffin'] test : 1184

Test accuracy: 87.3 %

## 4차 시도 (Rotation 증강 - rotation.py)

RandomRotation(15) augmentation 적용
batch_size = 16
lr=0.001
epochs = 10

sample: 4733 ['chihuahua', 'muffin'] test : 1184

Test accuracy: 88.4 %

## 5차 시도 (Gaussian Noise / Blur - noise.py)

batch_size = 16
lr=0.001
epochs = 10

sample: 4733 ['chihuahua', 'muffin'] test : 1184

gaussian (가우시안 노이즈 증강):

Test accuracy (clean): 84.5 %
Test accuracy (noisy eval): 85.1 %

blur (가우시안 블러 증강):

Test accuracy (clean): 86.0 %
