from dataset import get_data
import numpy as np
import matplotlib.pyplot as plt
X_train, X_test, Y_train, Y_test = get_data("./dataset/forestfires.csv")
names = ["bais", "X", "Y", "1th", "day", "FFMC", "DMC",
         "DC", "ISI", "temp", "RH", "wind", "rain", "area"]


def Error_loss(_X, _Y, _beta):
    return np.linalg.norm(_Y - _X @ _beta) ** 2


def Ridge_regression(_X, _Y, _lambda):
    return np.linalg.inv(_X.T @ _X + _lambda * np.identity(_X.shape[1])) @ _X.T @ _Y

#解析解直接计算
lambdas = np.arange(0.0, 200.0, 0.1)
error_tests = []
for lambda_ in lambdas:
    beta = Ridge_regression(X_train, Y_train, lambda_)
    error_tests.append(Error_loss(X_test, Y_test, beta))

# 可视化不同的lambda对test_error的影响
plt.figure()
plt.plot(lambdas, error_tests, label="train loss MSE")
plt.xlabel("lambda")
plt.ylabel("error_test")
plt.show()

#输出lambda=75时的误差
lambda_ = 75
beta = Ridge_regression(X_train, Y_train, lambda_)
error_test = Error_loss(X_test, Y_test, beta)
print("error test 75 =", error_test)

#输出lambda=90时的误差
lambda_ = 90
beta = Ridge_regression(X_train, Y_train, lambda_)
error_test = Error_loss(X_test, Y_test, beta)
print("error test 90 =", error_test)


# 可视化由大到小的beta
plt.figure()
sorted_idx = np.argsort(-beta)
sorted_names = [names[i] for i in sorted_idx]
sorted_values = [beta[i] for i in sorted_idx]
plt.bar(sorted_names, sorted_values)
plt.xlabel("Coefficient")
plt.ylabel("Value")
plt.show()
