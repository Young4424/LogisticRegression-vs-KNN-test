# logistic_regression_train_test.py
from LogisticRegression import LogisticRegression
from mnist_data import loadData
import numpy as np
import matplotlib.pyplot as plt

# MNIST 데이터 로드
x_train, y_train, x_test, y_test = loadData()

# 원-대-모두(one-vs-all) 방법을 위한 레이블 벡터 생성
y_train_digit = (y_train == 0).astype(int)  # 예시로 '0'을 대상으로 함
y_test_digit = (y_test == 0).astype(int)

# 모델 초기화 및 학습
lr = 0.00001
epochs = 300

model = LogisticRegression(lr, epochs)


# LR_test.py 파일 내에서 train 함수 호출 부분
losses = model.train(x_train, y_train_digit, x_test, y_test_digit)


# 손실 그래프 그리기
plt.plot(losses)
plt.xlabel('Epochs')
plt.ylabel('cost')
plt.title('Training cost over Epochs')
plt.show()
