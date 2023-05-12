from dataset import get_data
import numpy as np
import matplotlib.pyplot as plt
X_train, X_test, Y_train, Y_test = get_data("./dataset/forestfires.csv")
names = ["bais", "X", "Y", "1th", "day", "FFMC", "DMC",
         "DC", "ISI", "temp", "RH", "wind", "rain", "area"]


def Error_loss(_X, _Y, _beta):
    return np.linalg.norm(_Y - _X @ _beta) ** 2


def Lasso_regression(_X, _Y, _beta, _lambda):
    return np.linalg.norm(_Y - _X @ _beta, 2) ** 2 / 2 + _lambda * np.linalg.norm(_beta, 1)


def Grandient_MSE(_X, _Y, _beta, _lambda):
    return -1 / len(X_train) * X_train.T @ (Y_train - X_train @ beta) + _lambda * np.sign(beta)


# 梯度下降
step = 0.001
iteration = 7000
lambda_ = 1
beta = np.zeros((X_train.shape[1]))
draw_loss_train = []
for i in range(iteration):
    loss_train = Lasso_regression(X_train, Y_train, beta, lambda_)

    draw_loss_train.append(loss_train)

    beta = beta - step * Grandient_MSE(X_train, Y_train, beta, lambda_)

# 输出误差
error_test = Error_loss(X_test, Y_test, beta)
print("error test =", error_test)

# 可视化梯度下降过程
plt.figure()
plt.plot(range(len(draw_loss_train)), draw_loss_train)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.show()

# 可视化由大到小的beta
plt.figure()
sorted_idx = np.argsort(-beta)
sorted_names = [names[i] for i in sorted_idx]
sorted_values = [beta[i] for i in sorted_idx]
plt.bar(sorted_names, sorted_values)
plt.xlabel("Coefficient")
plt.ylabel("Value")
plt.show()
