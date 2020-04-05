import tensorflow as tf
import matplotlib.pyplot as plt


def cnn_fashion_mnist_exec():
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

    """
    conv2D layer는 채널을 명시한 shape의 데이터를 input으로 받기 때문에 채널의 수를 명시하도록 shape를 수정해야 함
    
    numpy.reshape() parameter에 들어있는 -1의 의미
    One shape dimension can be -1.
    In this case, the value is inferred from the length of the array and remaining dimensions.
    """
    train_x = train_x.reshape(-1, 28, 28, 1)  # 여기서 -1은 데이터 수를 의미
    test_x = test_x.reshape(-1, 28, 28, 1)
    # plt.figure(figsize=(10,10))
    # for c in range(9):
    #     plt.subplot(3,3, c+1)
    #     plt.imshow(train_x[c].reshape(28,28), cmap='gray')
    # plt.show()
    # print(train_y[:9])



def run_train():
    global train_x, train_y, model
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(input_shape=(28,28,1), kernel_size=(3,3), filters=32))
    model.add(tf.keras.layers.MaxPool2D(strides=(2,2)))
    model.add(tf.keras.layers.Conv2D(kernel_size=(3,3), filters=64))
    model.add(tf.keras.layers.MaxPool2D(strides=(2, 2)))
    model.add(tf.keras.layers.Conv2D(kernel_size=(3, 3), filters=128))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=128, activation='relu'))
    model.add(tf.keras.layers.Dropout(rate=0.3))
    model.add(tf.keras.layers.Dense(units=10, activation='softmax'))

    model.compile(optimizer=tf.keras.optimizers.Adam(),\
                  loss='sparse_categorical_crossentropy', \
                  metrics=['accuracy'])

    model.summary()

    model.fit(train_x, train_y, epochs=25, batch_size=128, validation_split=0.25)



def run_test():
    global test_x, test_y, model
    model.evaluate(test_x, test_y)


if __name__ == "__main__":
    cnn_fashion_mnist_exec()
