import numpy as np

class LogisticRegression:
    def __init__(self, lr=0.01, epochs=100, n_features=784):
        self.lr = lr  # 학습률
        self.epochs = epochs  # 학습 데이터를 반복한 횟수
        self.weights = np.zeros(n_features + 1)  # 가중치와 편향 초기화

    def _add_bias(self, X):
        # 입력 데이터에 편향을 추가
        bias = np.ones((X.shape[0], 1))
        return np.concatenate((bias, X), axis=1)

    def _sigmoid(self, z):
        # 시그모아드 함수 계산하기
        # 큰 z 값에 대한 오버플로 방지
        z = np.clip(z, -250, 250)
        return 1 / (1 + np.exp(-z))

    def cost(self, h, y):
        # 비용 함수 계산하기
        # h 값이 0이나 1이 되는 것을 방지
        h = np.clip(h, 1e-10, 1 - 1e-10)
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

    def train(self, X_train, y_train, X_test, y_test):
        # 로지스틱 회귀모델 학습
        X_train = self._add_bias(X_train)  
        losses = []  # 각 epoch 마다 loss값 저장하기 위한 리스트 

        for i in range(self.epochs):
            z = np.dot(X_train, self.weights)
            h = self._sigmoid(z)
            gradient = np.dot(X_train.T, (h - y_train)) / y_train.size
            self.weights -= self.lr * gradient

            loss = self.cost(h, y_train)
            losses.append(loss)  # 손실을 저장 

            if (i % 10 == 0):  # 선택적으로 매 10단계마다 손실을 출력
                print(f'Epoch {i}, loss: {loss}')

        # 테스트 세트에 대한 정확도를 계산
        test_accuracy = self.accuracy(X_test, y_test)
        print(f'\nTest accuracy: {test_accuracy * 100:.2f}%')

        return losses  # 학습이 끝난 후 손실 리스트를 반환

    def predict(self, X):
        #X의 샘플에 대한 클래스 레이블을 예측
        X = self._add_bias(X)  
        z = np.dot(X, self.weights)
        return np.round(self._sigmoid(z))

    def accuracy(self, X, y):
        # 모델의 정확도 계산
        predictions = self.predict(X)
        return np.mean(predictions == y)


