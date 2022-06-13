'''
Это решение одной из задач по ТВИМСу.
Для решения второй задачи необходимо задать переменные.
Для этого мы подготовим данные в первой задаче.
'''


'''
Сгенерируйте выборку из случайных величин с распределением Коши с плотностью
$$f(x) = \frac{1}{\pi \gamma \cdot \left(1 + \frac{(x - x_0)^2}{\gamma^2}\right)}$$
из выборки $u, u_{large} \sim U[0, 1]$. Возьмите параметры `gamma=10, x_0=20`.
'''
import numpy as np

#Решение:

rng = np.random.default_rng(0)
u = rng.uniform(0, 1, 50)
u_large = rng.uniform(0, 1, 10000)

gamma = 10
x_0 = 20

x = x_0 + gamma * np.tan(np.pi * (u - 1/2))
x_large = x_0 + gamma * np.tan(np.pi * (u_large - 1/2))

'''
Методом максимального правдоподобия оцените параметры $\gamma$ и $x_0$.
Явного решения найти не получится, поэтому предлагается воспользоваться алгоритмом градиентного подъема.
Инициализируйте значения единицами, а затем обновляйте значения параметров по формуле:
$$
\hat{\gamma}^{(t)} = \hat{\gamma}^{(t - 1)} + \eta \cdot \dfrac{\partial \ell}{\partial \gamma}
(\hat{\gamma}^{(t - 1)}, \hat{x_0}^{(t - 1)}) \\
$$

$$
\hat{x_0}^{(t)} = \hat{x_0}^{(t - 1)} + \eta \cdot \dfrac{\partial \ell}
{\partial x_0} (\hat{\gamma}^{(t - 1)}, \hat{x_0}^{(t - 1)})
$$
Поэкспериментируйте с гиперпараметром $\eta$ отвечающим за величину шага
(лучше брать его довольно маленьким, порядка 0.0001).
При желании вы можете воспользоваться другим итеративным методом (например методом Ньютона-Рафсона).
Прокомментируйте, насколько хорошо сработала численная оптимизация.
Сравните результаты, полученные на маленькой выборке и на большой.
'''
#Решение:
def grad_ascent(x, n_iters, step=0.0001):
    gamma_hat = 1
    x_0_hat = 1

    for i in range(n_iters):
        gamma_grad = -x.shape[0] / gamma_hat + 2 * np.sum((x - x_0_hat)**2 / (gamma_hat**3 + gamma_hat * (x - x_0_hat)**2))
        x_grad = 2 * np.sum((x - x_0_hat) / (gamma_hat**2 + (x - x_0_hat)**2))

        gamma_hat += step * gamma_grad
        x_0_hat += step * x_grad

    return gamma_hat, x_0_hat

print(grad_ascent(x, 10000, step=0.001))
print(grad_ascent(x_large, 10000, step=0.0001))