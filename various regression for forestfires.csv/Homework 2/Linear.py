from dataset import get_data
import numpy as np
import matplotlib.pyplot as plt
X_train, X_test, Y_train, Y_test = get_data("./dataset/forestfires.csv")
names = ["bais", "X", "Y", "1th", "day", "FFMC", "DMC",
         "DC", "ISI", "temp", "RH", "wind", "rain", "area"]


def Loss_MSE(_X, _Y, _beta):
    return np.linalg.norm(_Y - _X @ _beta) ** 2 / len(_X)


def Error_loss(_X, _Y, _beta):
    return np.linalg.norm(_Y - _X @ _beta) ** 2


def Grandient_MSE(_X, _Y, _beta):
    return -2 / len(_X) * _X.T @ (_Y - _X @ _beta)

#梯度下降
step = 0.001
iteration = 2000
beta = np.zeros(X_train.shape[1])
draw_loss_train = []
for i in range(iteration):
    draw_loss_train.append(Loss_MSE(X_train, Y_train, beta))
    beta = beta - step * Grandient_MSE(X_train, Y_train, beta)

#输出误差
train_loss_mse = Loss_MSE(X_train, Y_train, beta)
test_loss_mse = Loss_MSE(X_test, Y_test, beta)
error_test = Error_loss(X_test, Y_test, beta)
error_initial = Error_loss(X_test, Y_test, np.zeros(X_train.shape[1]))
error_math = Error_loss(X_test, Y_test, np.linalg.inv(X_train.T @ X_train) @ X_train.T @ Y_train)
print("train_loss_mse =", train_loss_mse)
print("test_loss_mse =", test_loss_mse)
print("error test =", error_test)
print("error test initial =", error_initial)
print("error test math =", error_math)

# 可视化梯度下降过程
plt.figure()
plt.plot(range(len(draw_loss_train)), draw_loss_train)
plt.xlabel('Iterations')
plt.ylabel('Loss_MSE')
plt.show()

#beta = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ Y_train
# 可视化由大到小的beta
plt.figure()
sorted_idx = np.argsort(-beta)
sorted_names = [names[i] for i in sorted_idx]
sorted_values = [beta[i] for i in sorted_idx]
plt.bar(sorted_names, sorted_values)
plt.xlabel("Coefficient")
plt.ylabel("Value")
plt.show()
