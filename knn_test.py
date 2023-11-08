import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from PIL import Image
from mnist_data import *
from knn import KNN


#load data
x_train, y_train, x_test, y_test = loadData()

x_train = x_train[:2000]
y_train = y_train[:2000]

# 테스트 데이터도 필요에 따라 줄임
x_test = x_test[:2000]
y_test = y_test[:2000]

# 모델 생성 및 학습
knn = KNN(k=6)
knn.fit(x_train, y_train)

# 모델 평가
accuracy = knn.evaluate(x_test, y_test)
print(f'Accuracy: {accuracy:.2%}')

