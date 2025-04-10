import math

import numpy as np
import matplotlib.pyplot as plt

k_abs = 100
w1 = 0.7
w2 = 0.3
Le = 4
Lo = 4
We = 2
Wo = 2
ve = 10
T0 = 2
a_max = 3


def test():
    predict_trajectory_primitive = np.array([[1.77936092e+02, 8.85789332e-02], [1.81046972e+02, 1.06106987e-01],
                                             [1.84155538e+02, 2.23489843e-01],
                                             [1.87254991e+02, 4.87991010e-01], [1.90338299e+02, 9.00450276e-01],
                                             [1.93404137e+02, 1.42785178e+00], [1.96458642e+02, 2.01777025e+00],
                                             [1.99512753e+02, 2.60975427e+00], [2.02577500e+02, 3.14339970e+00],
                                             [2.05659350e+02, 3.56612582e+00], [2.08757552e+02, 3.84365499e+00],
                                             [2.11865511e+02, 3.97365027e+00], [2.14976289e+02, 3.99961363e+00],
                                             [2.18087253e+02, 4.00000000e+00], [2.21198218e+02, 4.00000000e+00],
                                             [2.24309183e+02, 4.00000000e+00], [2.27420148e+02, 4.00000000e+00],
                                             [2.30531112e+02, 4.00000000e+00], [2.33642077e+02, 4.00000000e+00],
                                             [2.36753042e+02, 4.00000000e+00], [2.39864007e+02, 4.00000000e+00]])
    predict_heading = np.array([5.63437702e-03, 3.77431629e-02, 8.51317445e-02, 1.32982221e-01,
                                1.70357805e-01, 1.90781751e-01, 1.91457701e-01, 1.72395368e-01,
                                1.36315694e-01, 8.93390683e-02, 4.18022032e-02, 8.34606469e-03,
                                1.24197218e-04, -1.42749676e-16, 1.42749676e-16, 0.00000000e+00,
                                0.00000000e+00, -1.42749676e-16, 1.42749676e-16, 0.00000000e+00,
                                0.00000000e+00])

    obstacle_x = np.linspace(0, 40, 21)
    obstacle_y = np.full((21,), 4.01)
    temp_ego_x = predict_trajectory_primitive[0][0]
    for i in range(len(predict_trajectory_primitive)):
        predict_trajectory_primitive[i][0] -= temp_ego_x

    plt.figure(1, figsize=(6, 0.87))
    plt.cla()  # 清除当前figure
    plt.axis('off')
    plt.plot(predict_trajectory_primitive[:, 0], -predict_trajectory_primitive[:, 1], 'r--')  # 绘制虚线轨迹
    plt.scatter(predict_trajectory_primitive[:, 0], -predict_trajectory_primitive[:, 1], s=10, c='r')  # 绘制轨迹散点
    plt.plot(obstacle_x, -obstacle_y, 'y--')  # 绘制虚线轨迹
    plt.scatter(obstacle_x, -obstacle_y, s=10, c='y')  # 绘制轨迹散点
    plt.pause(1)

    rho = []
    for i in range(len(predict_trajectory_primitive)):
        Delta_s = predict_trajectory_primitive[i][0] - obstacle_x[i]
        Delta_l = predict_trajectory_primitive[i][1] - obstacle_y[i]
        Delta_v_s = 0
        Delta_v_l = 0
        Delta_a_s = 0
        Delta_a_l = 0

        Ss = 0.5 * (Le + Lo) + ve * T0 + (Delta_v_s ** 2) / (2 * a_max)
        Sl = 0.5 * (We + Wo) + (Delta_v_l ** 2) / (2 * a_max)

        # 定义 B1, B2 矩阵和 P1, P2 向量
        B1 = np.array([[(Ss - Delta_s) ** 2, 0],
                       [0, (Sl - Delta_l) ** 2]])

        B2 = np.array([[Ss ** 2, 0],
                       [0, Sl ** 2]])

        P1 = np.array([Delta_s, Delta_l])

        P2 = np.array([(1 - np.sign(Delta_a_s)) * Delta_s, (1 - np.sign(Delta_a_l)) * Delta_l])

        sqrt_det_B1 = np.sqrt(np.linalg.det(B1))
        sqrt_det_B2 = np.sqrt(np.linalg.det(B2))

        inv_B1 = np.linalg.inv(B1)
        inv_B2 = np.linalg.inv(B2)

        exp_term1 = np.exp(-0.5 * np.dot(P1, np.dot(inv_B1, P1.T)))
        exp_term2 = np.exp(-0.5 * np.dot(P2, np.dot(inv_B2, P2.T)))

        rho_i = (k_abs / (2 * np.pi)) * (w1 / sqrt_det_B1 * exp_term1 + w2 / sqrt_det_B2 * exp_term2)

        rho.append(rho_i)
    rho = np.array(rho)
    plt.figure(2, figsize=(6, 0.87))
    x = range(1, len(rho) + 1)  # 如果你的数据从1开始，可以使用 range(1, 22)，因为21个元素
    plt.plot(x, rho, 'r--')  # 绘制虚线轨迹
    plt.scatter(x, rho, s=10, c='r')  # 绘制轨迹散点

    risk_weight_list = []
    for i in x:
        risk_weight = 1 - math.exp(0.2 * (i - len(predict_trajectory_primitive)))
        risk_weight_list.append(risk_weight)
    risk_weight_list = np.array(risk_weight_list)
    plt.figure(3, figsize=(6, 0.87))
    plt.plot(x, risk_weight_list, 'r--')
    plt.scatter(x, risk_weight_list, s=10, c='r')

    plt.show()


if __name__ == "__main__":
    test()
