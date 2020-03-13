import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def run_train(x, y):
    global model, history
    model = tf.keras.Sequential([
                    tf.keras.layers.Dense(units=2, activation='sigmoid', input_shape=(2,)),
                    tf.keras.layers.Dense(units=1, activation='sigmoid')
    ])

    model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.3), loss='mse')
    model.summary()

    history = model.fit(x, y, epochs=2000, batch_size=1)


def run_test(x, y):
    global model
    for i in range(4):
        print('X: {}, Y: {}, output: {}'.format(x[i], y[i], model.predict(x)[i]))


def xorNet_exec():
    x = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
    y = np.array([[0], [1], [1], [0]])

    run_train(x, y)
    run_test(x, y)


if __name__ == "__main__":
    xorNet_exec()
    plt.plot(history.history['loss'])