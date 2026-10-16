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
