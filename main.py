import numpy as np

from constants import *
from train_and_test import train, test
from parse_data import processing_data


def print_weights(w1, w2):
    file = open(FILE_NAME_WEIGHTS, 'w')

    parse_w1, parse_w2 = "", ""

    for line in w1:
        for w in line:
            parse_w1 += str(w) + " "

    for line in w2:
        for w in line:
            parse_w2 += str(w) + " "

    file.write(parse_w1 + "\n" + parse_w2)
    file.close()


def print_bias(b1, b2):
    file = open(FILE_NAME_BIAS, 'w')

    parse_b1, parse_b2 = "", ""

    for b in b1:
        parse_b1 += str(b) + " "

    for b in b2:
        parse_b2 += str(b) + " "

    file.write(parse_b1 + "\n" + parse_b2)
    file.close()


if __name__ == '__main__':
    w1 = np.random.randn(INPUT_NEURONS, H_NEURONS)
    w2 = np.random.randn(H_NEURONS, OUT_NEURONS)
    b1 = np.random.randn(H_NEURONS)
    b2 = np.random.randn(OUT_NEURONS)

    (x_train, y_train), (x_test, y_test) = processing_data()

    find_w1, find_w2, find_b1, find_b2 = train(x_train, y_train, w1, w2, b1, b2)
    test_loss = test(x_test, y_test, find_w1, find_w2, find_b1, find_b2)
    print(f"Test Loss: {test_loss}")

    print_weights(find_w1, find_w2)
    print_bias(find_b1, find_b2)