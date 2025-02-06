import numpy as np
from parse_data import processing_data
from constants import *


w1 = np.random.randn(INPUT_NEURONS, H_NEURONS)
w2 = np.random.randn(H_NEURONS, OUT_NEURONS)
b1 = np.random.randn(H_NEURONS)
b2 = np.random.randn(OUT_NEURONS)

(x_train, y_train), (x_test, y_test) = processing_data()

def relu(t):
    return np.maximum(0, t)


def sigmoid_activation(t):
    return 1 / (1 + np.exp(-t))


def loss(y, z, n):
    return 1 / n * np.sum((y - z) ** 2)


def train(x_train, y_train):
    n = len(x_train)
    global w1, b1, w2, b2

    for epoch in range(EPOCHS):
        total_loss = float('inf')

        for x, y in zip(x_train, y_train):
            x = np.array(x, dtype=float)
            y = np.array(y, dtype=float)

            total_loss = float('inf')

            while total_loss > EPS:
                t1 = np.dot(x, w1) + b1
                h1 = relu(t1)
                t2 = np.dot(h1, w2) + b2
                z = sigmoid_activation(t2)

                total_loss = loss(y, z, n)
                print(total_loss)

                # для внешнего слоя
                grad_error = -2 * (y - z)
                dz_dt2 = z * (1 - z)
                dt2_dw2 = h1

                # рассчитываем градиент для внешнего слоя
                # –> для этого вычисляем произведение векторов h1
                # * градиент функции ошибки по предсказаниям для слоя w2 (вектор)

                grad_w2 = np.outer(dt2_dw2, grad_error * dz_dt2)
                grad_b2 = grad_error * dz_dt2

                # для внутреннего слоя
                d_error_dt2 = grad_error * dz_dt2
                dt2_dh1 = w2.T

                grad_h1 = np.dot(d_error_dt2, dt2_dh1)  # ошибка на выходе слоя h1
                grad_w1 = np.outer(x, grad_h1 * (t1 > 0))  # градиенты для w1
                grad_b1 = grad_h1 * (t1 > 0)  # градиенты для b1

                w1 -= LEARNING_RATE * grad_w1
                b1 -= LEARNING_RATE * grad_b1.sum(axis=0)
                w2 -= LEARNING_RATE * grad_w2
                b2 -= LEARNING_RATE * grad_b2.sum(axis=0)

        print(f"Epoch {epoch+1}/{EPOCHS}, Total Loss: {total_loss}")


# TODO сделать метод, которые будет анализировать обученная нейросеть


if __name__ == '__main__':
    train(x_train, y_train)