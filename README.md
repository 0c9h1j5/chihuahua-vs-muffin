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

