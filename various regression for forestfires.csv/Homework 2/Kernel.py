from dataset import get_data
import numpy as np
import matplotlib.pyplot as plt
X_train, X_test, Y_train, Y_test = get_data("./dataset/forestfires.csv")
names = ["bais", "X", "Y", "1th", "day", "FFMC", "DMC",
         "DC", "ISI", "temp", "RH", "wind", "rain", "area"]


def Error_loss(_Y_test, _Y_test_predict):
    return np.linalg.norm(_Y_test - _Y_test_predict) ** 2


def RBF_kernel(x1, x2, _gamma):
    return np.exp(-_gamma * np.linalg.norm(x1 - x2) ** 2)

#解析解直接计算
sigma_ = 0.01
lambda_ = 5
K = np.zeros((X_train.shape[0], X_train.shape[0]))
for i in range(X_train.shape[0]):
    for j in range(X_train.shape[0]):
        K[i, j] = RBF_kernel(X_train[i, :], X_train[j, :], sigma_)
c = np.linalg.inv(K + lambda_ * np.identity(X_train.shape[0])) @ Y_train
N_test = X_test.shape[0]
Y_test_predict = np.zeros(N_test)
for i in range(N_test):
    Ki = np.zeros(X_train.shape[0])
    for j in range(X_train.shape[0]):
        Ki[j] = RBF_kernel(X_test[i, :], X_train[j, :], sigma_)
    Y_test_predict[i] = (Ki) @ c

#输出误差
error_test = Error_loss(Y_test, Y_test_predict)
print("error test =", error_test)

#可视化c
plt.figure()
plt.bar(np.arange(X_train.shape[0]), c)
plt.xlabel('Index')
plt.ylabel('value')
plt.show()

#从大到小可视化c
plt.figure()
sorted_idx = np.argsort(-c)  # 从大到小排序
plt.bar(np.arange(X_train.shape[0]), c[sorted_idx])
plt.xlabel('Index')
plt.ylabel('value')
plt.show()