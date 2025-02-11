import numpy as np
from constants import *

def relu(t):
    """
    Функция активации ReLU

    :param t: векторное умножение входного параметра x на вес w1 и добавление смещения b1
    :return: возвращает t, если t > 0, иначе возвращает 0
    """
    return np.maximum(0, t)


def sigmoid_activation(t):
    """
    Функция активации сигмоиды

    :param t: входное значение или массив
    :return: возвращает значение сигмоидной функции: 1 / (1 + exp(-t))
    """
    return 1 / (1 + np.exp(-t))


def sigmoid_derivative(z):
    """
    Производная сигмоиды

    :param z: результат функции сигмоиды
    :return: производная сигмоиды: z * (1 - z)
    """
    return sigmoid_activation(z) * (1 - sigmoid_activation(z))


def loss(y, z):
    """
    Квадратичная функция потерь

    :param y: эталонное значение (истинное из выборки)
    :param z: предполагаемое значение нейросетью
    :return: квадрат разницы между эталонным и предполагаемым значениями
    """
    return (y - z) ** 2


def train(x_train, y_train, w1, w2, b1, b2):
    """
    Обучение нейросети методом градиентного спуска с минимизацией среднеквадратичной ошибки

    :param x_train: входные данные тренировочной выборки
    :param y_train: эталонные значения для входных данных
    :param w1: матрица весов для связей между входным и скрытым слоями
    :param w2: матрица весов для связей между скрытым и выходным слоями
    :param b1: вектор смещений для скрытого слоя
    :param b2: вектор смещений для выходного слоя

    :return: обновленные веса (w1, w2) и смещения (b1, b2)
    """

    n = len(x_train)
    total_loss = float('inf')
    epoch = 0

    while (epoch < EPOCHS) and (total_loss > EPS):
        total_loss = 0
        agr_grad_w1 = np.zeros_like(w1)
        agr_grad_w2 = np.zeros_like(w2)
        agr_grad_b1 = np.zeros_like(b1)
        agr_grad_b2 = np.zeros_like(b2)

        for x, y in zip(x_train, y_train):
            x = np.array(x, dtype=float)
            y = np.array(y, dtype=float)

            t1 = np.dot(x, w1) + b1
            h1 = relu(t1)
            t2 = np.dot(h1, w2) + b2
            z = sigmoid_activation(t2)

            total_loss += loss(y, z)

            # для внешнего слоя
            grad_error = -2 * (y - z)
            dz_dt2 = sigmoid_derivative(z)
            dt2_dw2 = h1

            # рассчитываем градиент для внешнего слоя
            # –> для этого вычисляем произведение векторов h1 и градиента функции ошибки
            # по предполагаемым значениям параметров в слое w2 (результат —> вектор)
            grad_w2 = np.outer(dt2_dw2, grad_error * dz_dt2)
            grad_b2 = grad_error * dz_dt2

            # для внутреннего слоя
            d_error_dt2 = grad_error * dz_dt2
            dt2_dh1 = w2.T
            grad_h1 = np.dot(d_error_dt2, dt2_dh1)
            grad_w1 = np.outer(x, grad_h1 * (t1 > 0))  # (t1 > 0) — производная relu
            grad_b1 = grad_h1 * (t1 > 0)

            agr_grad_w1 += grad_w1
            agr_grad_w2 += grad_w2
            agr_grad_b1 += grad_b1
            agr_grad_b2 += grad_b2

        agr_grad_w1 /= n
        agr_grad_b1 /= n
        agr_grad_w2 /= n
        agr_grad_b2 /= n

        w1 -= LEARNING_RATE * agr_grad_w1
        b1 -= LEARNING_RATE * agr_grad_b1
        w2 -= LEARNING_RATE * agr_grad_w2
        b2 -= LEARNING_RATE * agr_grad_b2

        epoch += 1
        total_loss /= n
        print(f"Epoch {epoch}/{EPOCHS}, Total Loss: {total_loss}")
    return w1, w2, b1, b2


def test(x_test, y_test, new_w1, new_w2, new_b1, new_b2):
    """
    Тестирование обученной нейросети

    :param x_test: входные данные тестовой выборки
    :param y_test: эталонные значения для входных данных тестовой выборки
    :param new_w1: обновленная матрица весов между входными нейронами и нейронами скрытого слоя
    :param new_w2: обновленная матрица весов между нейронами скрытого слоя и выходным нейроном
    :param new_b1: обновленный вектор смещений для скрытого слоя
    :param new_b2: обновленный вектор смещений для выходного слоя

    :return: возвращает среднеквадратичную ошибку на тестовой выборке
    """
    test_total_loss = 0
    n = len(x_test)

    for x, y in zip(x_test, y_test):
        x = np.array(x, dtype=float)
        y = np.array(y, dtype=float)

        new_t1 = np.dot(x, new_w1) + new_b1
        new_h1 = relu(new_t1)
        new_t2 = np.dot(new_h1, new_w2) + new_b2
        new_z = sigmoid_activation(new_t2)

        test_total_loss += loss(y, new_z)

    test_total_loss /= n
    return test_total_loss