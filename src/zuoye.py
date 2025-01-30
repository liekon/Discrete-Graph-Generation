""" # 导入必要的库
import numpy as np  # 用于数值计算
import matplotlib.pyplot as plt  # 用于绘制图形
from numpy.polynomial.polynomial import Polynomial  # 用于多项式拟合


# 定义龙格函数
def runge_function(x):
    return 1 / (1 + 25 * x**2)


# 第一题：多项式插值（等距节点）
def polynomial_interpolation_uniform(n):
    # 生成等距节点
    x_nodes = np.linspace(-1, 1, n)
    y_nodes = runge_function(x_nodes)

    # 构造插值多项式
    coefficients = np.polynomial.polynomial.polyfit(x_nodes, y_nodes, n - 1)
    poly = Polynomial(coefficients)

    # 绘制插值函数和原函数
    x_plot = np.linspace(-1, 1, 500)
    y_plot = runge_function(x_plot)
    y_interp = poly(x_plot)

    plt.figure()
    plt.plot(x_plot, y_plot, label='f(x) (Original function)')
    plt.plot(x_plot, y_interp, label=f'Interpolating Polynomial (n={n})')
    plt.scatter(x_nodes, y_nodes, color='red', label='Interpolation nodes')
    plt.title(f"Polynomial Interpolation (Uniform nodes, n={n})")
    plt.legend()
    plt.grid()
    plt.savefig(f"polynomial_interpolation_uniform_n{n}.png")  # 保存图片


# 第二题：分段线性插值
def piecewise_linear_interpolation(n):
    # 生成等距节点
    x_nodes = np.linspace(-1, 1, n)
    y_nodes = runge_function(x_nodes)

    # 绘制插值函数和原函数
    x_plot = np.linspace(-1, 1, 500)
    y_plot = runge_function(x_plot)

    plt.figure()
    plt.plot(x_plot, y_plot, label='f(x) (Original function)')
    plt.plot(x_nodes, y_nodes, label='Piecewise Linear Interpolation', color='orange')
    plt.scatter(x_nodes, y_nodes, color='red', label='Interpolation nodes')
    plt.title(f"Piecewise Linear Interpolation (n={n})")
    plt.legend()
    plt.grid()
    plt.savefig(f"piecewise_linear_interpolation_n{n}.png")  # 保存图片


# 第三题：最小二乘拟合
def least_squares_fit(n, degree):
    # 生成等距节点
    x_nodes = np.linspace(-1, 1, n)
    y_nodes = runge_function(x_nodes)

    # 进行最小二乘拟合
    coefficients = np.polynomial.polynomial.polyfit(x_nodes, y_nodes, degree)
    poly = Polynomial(coefficients)

    # 绘制拟合函数和原函数
    x_plot = np.linspace(-1, 1, 500)
    y_plot = runge_function(x_plot)
    y_fit = poly(x_plot)

    plt.figure()
    plt.plot(x_plot, y_plot, label='f(x) (Original function)')
    plt.plot(x_plot, y_fit, label=f'Least Squares Fit (degree={degree})', color='green')
    plt.scatter(x_nodes, y_nodes, color='red', label='Fit nodes')
    plt.title(f"Least Squares Fit (n={n}, degree={degree})")
    plt.legend()
    plt.grid()
    plt.savefig(f"least_squares_fit_n{n}_degree{degree}.png")  # 保存图片

    # 打印拟合多项式
    print(f"n={n}, degree={degree} Least Squares Fit Polynomial:")
    print(poly)


# 第四题：多项式插值（Chebyshev节点）
def polynomial_interpolation_chebyshev(n):
    # 生成 Chebyshev 节点
    x_nodes = np.cos((2 * np.arange(n) + 1) / (2 * n) * np.pi)
    y_nodes = runge_function(x_nodes)

    # 构造插值多项式
    coefficients = np.polynomial.polynomial.polyfit(x_nodes, y_nodes, n - 1)
    poly = Polynomial(coefficients)

    # 绘制插值函数和原函数
    x_plot = np.linspace(-1, 1, 500)
    y_plot = runge_function(x_plot)
    y_interp = poly(x_plot)

    plt.figure()
    plt.plot(x_plot, y_plot, label='f(x) (Original function)')
    plt.plot(x_plot, y_interp, label=f'Interpolating Polynomial (n={n}, Chebyshev)')
    plt.scatter(x_nodes, y_nodes, color='red', label='Interpolation nodes')
    plt.title(f"Polynomial Interpolation (Chebyshev nodes, n={n})")
    plt.legend()
    plt.grid()
    plt.savefig(f"polynomial_interpolation_chebyshev_n{n}.png")  # 保存图片


# 第五题：对比分析
def compare_approximations():
    ns = [10, 20]
    for n in ns:
        print(f"\n=== Comparison for n={n} ===")
        print("1. Polynomial Interpolation (Uniform nodes)")
        polynomial_interpolation_uniform(n)

        print("2. Piecewise Linear Interpolation")
        piecewise_linear_interpolation(n)

        print("3. Least Squares Fit (degree=3)")
        least_squares_fit(n, 3)

        print("4. Least Squares Fit (degree=5)")
        least_squares_fit(n, 5)

        print("5. Polynomial Interpolation (Chebyshev nodes)")
        polynomial_interpolation_chebyshev(n)


# 主程序：运行所有题目
if __name__ == "__main__":
    print("Question 1: Polynomial Interpolation (Uniform nodes)")
    polynomial_interpolation_uniform(10)
    polynomial_interpolation_uniform(20)

    print("Question 2: Piecewise Linear Interpolation")
    piecewise_linear_interpolation(10)
    piecewise_linear_interpolation(20)

    print("Question 3: Least Squares Fit")
    least_squares_fit(10, 3)
    least_squares_fit(20, 3)
    least_squares_fit(10, 5)
    least_squares_fit(20, 5)

    print("Question 4: Polynomial Interpolation (Chebyshev nodes)")
    polynomial_interpolation_chebyshev(10)
    polynomial_interpolation_chebyshev(20)

    print("Question 5: Comparison")
    compare_approximations()
 """

# 导入必要的库
import numpy as np
import matplotlib.pyplot as plt


# 定义龙格函数
def runge_function(x):
    return 1 / (1 + 25 * x**2)


# 拉格朗日插值法实现
def lagrange_interpolation(x_nodes, y_nodes, x_eval):
    n = len(x_nodes)
    y_eval = np.zeros_like(x_eval)

    for i in range(n):
        # 构造拉格朗日基函数
        li = np.ones_like(x_eval)
        for j in range(n):
            if i != j:
                li *= (x_eval - x_nodes[j]) / (x_nodes[i] - x_nodes[j])
        # 计算插值结果
        y_eval += y_nodes[i] * li
    return y_eval


# 最小二乘拟合实现
def least_squares_fit(x_nodes, y_nodes, degree):
    # 构造 Vandermonde 矩阵
    A = np.vander(x_nodes, degree + 1, increasing=True)
    # 求解正规方程 A.T @ A @ coeffs = A.T @ y
    coeffs = np.linalg.solve(A.T @ A, A.T @ y_nodes)
    return coeffs


# 计算多项式值
def evaluate_polynomial(coeffs, x):
    y = np.zeros_like(x)
    for i, c in enumerate(coeffs):
        y += c * x**i
    return y

# 第一题：多项式插值（等距节点，使用拉格朗日插值法）
def polynomial_interpolation_uniform(n):
    # 等距节点
    x_nodes = np.linspace(-1, 1, n)
    y_nodes = runge_function(x_nodes)

    # 评估点
    x_plot = np.linspace(-1, 1, 500)
    y_plot = runge_function(x_plot)

    # 拉格朗日插值
    y_interp = lagrange_interpolation(x_nodes, y_nodes, x_plot)

    plt.figure()
    plt.plot(x_plot, y_plot, label='f(x) (Original function)')
    plt.plot(x_plot, y_interp, label=f'Interpolating Polynomial (n={n})')
    plt.scatter(x_nodes, y_nodes, color='red', label='Interpolation nodes')
    plt.title(f"Polynomial Interpolation (Uniform nodes, n={n})")
    plt.legend()
    plt.grid()
    plt.savefig(f"polynomial_interpolation_uniform_n{n}1.png")  # 保存图片



# 第二题：分段线性插值
def piecewise_linear_interpolation(n):
    # 等距节点
    x_nodes = np.linspace(-1, 1, n)
    y_nodes = runge_function(x_nodes)

    # 评估点
    x_plot = np.linspace(-1, 1, 500)
    y_plot = runge_function(x_plot)

    # 分段线性插值
    y_interp = np.interp(x_plot, x_nodes, y_nodes)

    plt.figure()
    plt.plot(x_plot, y_plot, label='f(x) (Original function)')
    plt.plot(x_nodes, y_nodes, label='Piecewise Linear Interpolation', color='orange')
    plt.scatter(x_nodes, y_nodes, color='red', label='Interpolation nodes')
    plt.title(f"Piecewise Linear Interpolation (n={n})")
    plt.legend()
    plt.grid()
    plt.savefig(f"piecewise_linear_interpolation_n{n}1.png")  # 保存图片


# 第三题：最小二乘拟合
def least_squares_fitting(n, degree):
    # 等距节点
    x_nodes = np.linspace(-1, 1, n)
    y_nodes = runge_function(x_nodes)

    # 评估点
    x_plot = np.linspace(-1, 1, 500)
    y_plot = runge_function(x_plot)

    # 最小二乘拟合
    coeffs = least_squares_fit(x_nodes, y_nodes, degree)
    y_fit = evaluate_polynomial(coeffs, x_plot)

    plt.figure()
    plt.plot(x_plot, y_plot, label='f(x) (Original function)')
    plt.plot(x_plot, y_fit, label=f'Least Squares Fit (degree={degree})', color='green')
    plt.scatter(x_nodes, y_nodes, color='red', label='Fit nodes')
    plt.title(f"Least Squares Fit (n={n}, degree={degree})")
    plt.legend()
    plt.grid()
    plt.savefig(f"least_squares_fit_n{n}_degree{degree}1.png")  # 保存图片

    # 打印拟合多项式
    print(f"n={n}, degree={degree} Least Squares Fit Polynomial Coefficients:")
    print(coeffs)


# 第四题：多项式插值（Chebyshev节点）
def polynomial_interpolation_chebyshev(n):
    # Chebyshev 节点
    x_nodes = np.cos((2 * np.arange(n) + 1) / (2 * n) * np.pi)
    y_nodes = runge_function(x_nodes)

    # 评估点
    x_plot = np.linspace(-1, 1, 500)
    y_plot = runge_function(x_plot)

    # 拉格朗日插值
    y_interp = lagrange_interpolation(x_nodes, y_nodes, x_plot)

    plt.figure()
    plt.plot(x_plot, y_plot, label='f(x) (Original function)')
    plt.plot(x_plot, y_interp, label=f'Interpolating Polynomial (n={n}, Chebyshev)')
    plt.scatter(x_nodes, y_nodes, color='red', label='Interpolation nodes')
    plt.title(f"Polynomial Interpolation (Chebyshev nodes, n={n})")
    plt.legend()
    plt.grid()
    plt.savefig(f"polynomial_interpolation_chebyshev_n{n}1.png")  # 保存图片

# 主程序：运行所有题目
if __name__ == "__main__":
    print("Question 1: Polynomial Interpolation (Uniform nodes)")
    polynomial_interpolation_uniform(10)
    polynomial_interpolation_uniform(20)

    print("Question 2: Piecewise Linear Interpolation")
    piecewise_linear_interpolation(10)
    piecewise_linear_interpolation(20)

    print("Question 3: Least Squares Fit")
    least_squares_fitting(10, 3)
    least_squares_fitting(20, 3)
    least_squares_fitting(10, 5)
    least_squares_fitting(20, 5)

    print("Question 4: Polynomial Interpolation (Chebyshev nodes)")
    polynomial_interpolation_chebyshev(10)
    polynomial_interpolation_chebyshev(20)
