import numpy as np
import time
from mnist_data import loadData  

class KNN:
    def __init__(self, k=5):
        self._k = k  # KNN에서 사용할 최근접 이웃의 수 k를 초기화
        self._fit_time = None  # fit 메소드가 실행된 시간을 저장하기 위한 변수 초기화

    def fit(self, x_train, y_train):
        start_time = time.time()  # 학습을 시작하기 전의 시간을 기록
        self._x_train = x_train[:2000]  # 훈련 데이터 중 2000개만 사용
        self._y_train = y_train[:2000]  # 훈련 라벨 중 2000개만 사용
        self._fit_time = time.time() - start_time  # fit 메소드의 실행 시간을 계산

    def predict(self, x_test, method="majority"):
        start_time = time.time()  # 예측을 시작하기 전의 시간을 기록

        # 각 테스트 샘플과 모든 훈련 샘플 간의 유클리드 거리를 계산
        distances = np.sqrt(((x_test[:, np.newaxis] - self._x_train) ** 2).sum(axis=2))
        
        end_time = time.time()  # 예측이 끝난 후의 시간을 기록
        print(f'Prediction time: {end_time - start_time:.2f} seconds')  # 예측에 걸린 시간을 출력


        # 'majority' 방식으로 예측할지, 'weighted_majority' 방식으로 예측할지 결정한다.
        if method == "majority":
            predictions = self._predict_majority(distances)
        elif method == "weighted_majority":
            predictions = self._predict_weighted_majority(distances)
        else:
            raise ValueError("Invalid method")  # 잘못된 예측 방식을 입력했을 경우 에러를 발생시킴
        
        return predictions

    def _predict_majority(self, distances):
        predictions = []
        for i in range(distances.shape[0]):  
            neighbors = np.argsort(distances[i])[:self._k]  # 각 테스트 샘플에 대해 가장 가까운 k개의 이웃을 찾는다.

            votes = np.zeros(10)  # 각 숫자 클래스(0부터 9까지)에 대한 투표 수를 담을 배열을 생성
            
            for neighbor in neighbors:  
                votes[self._y_train[neighbor]] += 1  # 가까운 이웃들을 순회하며 해당 이웃의 클래스에 투표한다.
            predictions.append(np.argmax(votes))  # 가장 많은 투표를 받은 클래스를 예측 결과로 추가
        return predictions

    def _predict_weighted_majority(self, distances):
        predictions = []
        for i in range(distances.shape[0]):  
            neighbors = np.argsort(distances[i])[:self._k]  # 가까운 이웃들을 순회하며 해당 이웃의 클래스에 투표한다.
            votes = np.zeros(10)  # 각 숫자 클래스(0부터 9까지)에 대한 투표 수를 담을 배열을 생성
            
            for neighbor in neighbors: 
                # 거리의 역수를 가중치로 사용하여 해당 이웃의 클래스에 투표합니다 (거리가 0이 되는 것을 방지하기 위해 1e-5를 더함)
                votes[self._y_train[neighbor]] += 1 / (distances[i][neighbor] + 1e-5)
            predictions.append(np.argmax(votes))  # 가장 많은 가중 투표를 받은 클래스를 예측 결과로 추가
        return predictions

    def evaluate(self, x_test, y_test, method="majority"):
        predictions = self.predict(x_test, method)  # 테스트 데이터에 대해 예측을 수행
        accuracy = np.mean(predictions == y_test)  # 예측 정확도를 계산
        return accuracy  # 정확도를 반환
