import numpy as np
from parse_data import processing_data
from constants import *

w1 = np.random.rand(INPUT_NEURONS, H_NEURONS)
w2 = np.random.rand(H_NEURONS, OUT_NEURONS)
b1 = np.random.rand(H_NEURONS)
b2 = np.random.rand(OUT_NEURONS)

(x_train, y_train), (x_test, y_test) = processing_data()

loss_check = 0.1

def relu(t):
    return np.maximum(0, t)

def sigmoid_derivative(x):
    return sigmoid_activation(x) * (1 - sigmoid_activation(x))

def sigmoid_activation(t):
    return 1 / (1 + np.exp(-t))


def loss(y, z):
    """
    Квадратичная функция потерь

    :param y: эталонное значение из тренировочной выборки сета
    :param z: предполагаемое значение нейросети —> sigmoid(dot(h1, w2) + b2)
    :return: возвращает разницу между значениями
    """
    return (y - z) ** 2


# Обучение
def train(x_train, y_train):
    n = len(x_train)
    global w1, b1, w2, b2

    total_loss = float('inf')
    epoch = 0

    while (epoch < EPOCHS) and (total_loss > loss_check):
        total_loss = 0
        for x, y in zip(x_train, y_train):
            x = np.array(x, dtype=float)
            print("x: ", x)
            print("w1: ", w1)
            y = np.array(y, dtype=float)

            t1 = np.dot(x, w1) + b1
            h1 = relu(t1)
            t2 = np.dot(h1, w2) + b2
            z = sigmoid_activation(t2)

            total_loss += loss(y, z)

            # для внешнего слоя
            grad_error = -2 * (y - z)
            dz_dt2 = z * (1 - z)  # Производная сигмоиды
            dt2_dw2 = h1

            # рассчитываем градиент для внешнего слоя
            # –> для этого вычисляем произведение векторов h1
            # * градиент функции ошибки по предсказаниям для слоя w2 (вектор)

            grad_w2 = np.outer(dt2_dw2, grad_error * dz_dt2)
            grad_b2 = grad_error * dz_dt2

            # для внутреннего слоя
            d_error_dt2 = grad_error * dz_dt2
            dt2_dh1 = w2.T
            grad_h1 = np.dot(d_error_dt2, dt2_dh1)
            grad_w1 = np.outer(x, grad_h1 * (t1 > 0))  # Производная ReLU
            grad_b1 = grad_h1 * (t1 > 0)

            w1 -= LEARNING_RATE * grad_w1
            b1 -= LEARNING_RATE * grad_b1
            w2 -= LEARNING_RATE * grad_w2
            b2 -= LEARNING_RATE * grad_b2

        epoch += 1
        total_loss /= n
        print(f"Epoch {epoch+1}/{EPOCHS}, Total Loss: {total_loss}")

if __name__ == '__main__':
    train(x_train, y_train)