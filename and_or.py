import math
import numpy as np
import tensorflow as tf


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def train(x, y):
    global w, b
    LEARNING_RATE = 0.1
    w = tf.random.normal([2], 0, 1)
    b = tf.random.normal([1], 0, 1)

    # epoch = 2000, batch size = 1
    for i in range(2000):
        losses = []
        for j in range(4):
            output = sigmoid(np.sum(x[j] * w) + b)
            loss = y[j][0] - output
            w += x[j] * LEARNING_RATE * loss
            b += LEARNING_RATE * loss
            losses.append(loss)

        if i % 200 == 199:
            print('epoch {}, loss: {}'.format(i + 1, np.mean(losses)))


def test(x, y):
    print('\ntest results')
    for i in range(4):
        print('X: {}, Y: {}, output: {}'.format(x[i], y[i], sigmoid(np.sum(x[i]*w) +b)))


def andNet_train_and_test():
    x = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
    y = np.array([[1], [0], [0], [0]])

    train(x, y)
    test(x, y)


def orNet_train_and_test():
    x = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
    y = np.array([[1], [1], [1], [0]])

    train(x, y)
    test(x, y)


if __name__ == '__main__':
    andNet_train_and_test()
    orNet_train_and_test()