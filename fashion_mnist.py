import tensorflow as tf
import matplotlib.pyplot as plt


def fashion_mnist_exec():
    load_data()
    preprocess_data()
    run_train()
    run_test()



def load_data():
    global train_x, train_y, test_x, test_y
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_x, train_y), (test_x, test_y) = fashion_mnist.load_data()
    # print(len(train_x), len(test_x))
    # plt.imshow(train_x[0], cmap='gray')
    # plt.colorbar()
    # plt.show()
    # print(train_y[0])



def preprocess_data():
    """
    min-max normalization
    """
    global train_x, train_y, test_x, test_y
    train_x = train_x / 255.0
    test_x = test_x / 255.0



def run_train():
    global train_x, train_y, model
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
    model.add(tf.keras.layers.Dense(units=128, activation='relu'))
    model.add(tf.keras.layers.Dense(units=10, activation='softmax'))

    model.compile(optimizer=tf.keras.optimizers.Adam(),\
                  loss='sparse_categorical_crossentropy', \
                  metrics=['accuracy'])

    model.summary()

    history = model.fit(train_x, train_y, epochs=30, batch_size=256, validation_split=0.25,\
                        callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, monitor='val_loss')])

    # visualization
    plt.figure(figsize=(12,4))

    plt.subplot(1,2,1)
    plt.plot(history.history['loss'], 'b-', label='loss')
    plt.plot(history.history['val_loss'], 'r--', label='val_loss')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(history.history['accuracy'], 'g-', label='accuracy')
    plt.plot(history.history['val_accuracy'], 'k--', label='val_accuracy')
    plt.ylim(0.7, 1)
    plt.xlabel('Epoch')
    plt.legend()

    plt.show()



def run_test():
    global test_x, test_y, model
    model.evaluate(test_x, test_y)


if __name__ == "__main__":
    fashion_mnist_exec()
