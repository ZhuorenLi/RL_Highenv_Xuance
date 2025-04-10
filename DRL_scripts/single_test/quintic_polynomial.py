import numpy as np
import matplotlib.pyplot as plt


def solve_quintic_polynomial_coeffs(tf, x_0, x_tf, dx_0, dx_tf, ddx_0, ddx_tf):

    a0 = x_0
    a1 = dx_0
    a2 = ddx_0 / 2.0

    A = np.array([[tf ** 3, tf ** 4, tf ** 5],
                  [3 * tf ** 2, 4 * tf ** 3, 5 * tf ** 4],
                  [6 * tf, 12 * tf ** 2, 20 * tf ** 3]])
    b = np.array([x_tf - a2 * tf - a1 * tf - a0, dx_tf - a1 - 2 * a2 * tf, ddx_tf - 2 * a2])
    x = np.linalg.solve(A, b)

    a3 = x[0]
    a4 = x[1]
    a5 = x[2]

    t_list = np.linspace(0, tf, int(tf / 0.2) + 1)
    x_list = []
    v_list = []
    a_list = []
    for t in t_list:
        x_t = a5 * (t ** 5) + a4 * (t ** 4) + a3 * (t ** 3) + a2 * (t ** 2) + a1 * t + a0
        x_list.append(x_t)

        v_t = 5 * a5 * (t ** 4) + 4 * a4 * (t ** 3) + 3 * a3 * (t ** 2) + 2 * a2 * t + a1
        v_list.append(v_t)

        a_t = 20 * a5 * (t ** 3) + 12 * a4 * (t ** 2) + 6 * a3 * t + 2 * a2
        a_list.append(a_t)

    x_list = np.array(x_list)
    v_list = np.array(v_list)
    a_list = np.array(a_list)
    plt.figure()
    plt.plot(t_list, v_list, 'r--')
    plt.scatter(t_list, v_list, s=10, c='r')
    plt.show()

    return x_list, v_list, a_list

if __name__ == "__main__":

    # Example usage:
    x_t0 = 10.0    # 初始位置
    x_tf = 50.0    # RL来输出终点位置
    dx_t0 = 10      # 初始速度
    dx_tf = dx_t0   # 终点速度
    ddx_t0 = 0.5   # 初始加速度
    ddx_tf = 0     # 终点加速度

    tf = 4.0

    x_, x_v_, x_a_ = solve_quintic_polynomial_coeffs(tf, x_t0, x_tf, dx_t0, dx_tf, ddx_t0, ddx_tf)

    y_t0 = 0.0
    y_tf = 4.0
    dy_t0 = 0
    dy_tf = 0
    ddy_t0 = 0
    ddy_tf = 0

    # y_, y_v_, y_a_ = solve_quintic_polynomial_coeffs(tf, y_t0, y_tf, dy_t0, dy_tf, ddy_t0, ddy_tf)

    # plt.figure(figsize=(6, 0.87))
    # plt.plot(x_, y_, 'r--')
    # plt.scatter(x_, y_, s=10, c='r')
    # plt.show()
