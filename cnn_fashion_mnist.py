import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np


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
    image augmentation
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

    # 이미지 보강
    image_augmentation()



def image_augmentation():
    global train_x, train_y
    image_generator = ImageDataGenerator(rotation_range=10,
                                         zoom_range=0.10,
                                         shear_range=0.5,
                                         width_shift_range=0.10,
                                         height_shift_range=0.10,
                                         horizontal_flip=True,
                                         vertical_flip=False)
    augment_size = 30000

    # random하게 augment_size 만큼의 sample data를 선택함
    randidx = np.random.randint(train_x.shape[0], size=augment_size)
    x_selected = train_x[randidx].copy()
    y_selected = train_y[randidx].copy()

    # image_generator.flow()는 iterator를 return하며, iterator의 element 갯수는 data_size / batch_size 만큼 임
    x_augmented = image_generator.flow(x_selected, np.zeros(augment_size), batch_size=augment_size, shuffle=False).next()[0]
    train_x = np.concatenate((train_x, x_augmented))
    train_y = np.concatenate((train_y, y_selected))



def run_train():
    """
    VGGNet 스타일로 레이어를 충분히 쌓고 이미지 보강(Image Augmentation)을 통해 성능을 향상 시켜봄
    """
    global train_x, train_y, model
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(input_shape=(28,28,1), kernel_size=(3,3), filters=32, padding='same', activation='relu'))
    model.add(tf.keras.layers.Conv2D(kernel_size=(3,3), filters=64, padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(strides=(2,2)))
    model.add(tf.keras.layers.Dropout(rate=0.5))
    model.add(tf.keras.layers.Conv2D(kernel_size=(3,3), filters=128, padding='same', activation='relu'))
    model.add(tf.keras.layers.Conv2D(kernel_size=(3,3), filters=256, padding='valid', activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(strides=(2, 2)))
    model.add(tf.keras.layers.Dropout(rate=0.5))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=512, activation='relu'))
    model.add(tf.keras.layers.Dropout(rate=0.5))
    model.add(tf.keras.layers.Dense(units=256, activation='relu'))
    model.add(tf.keras.layers.Dropout(rate=0.5))
    model.add(tf.keras.layers.Dense(units=10, activation='softmax'))

    model.compile(optimizer=tf.keras.optimizers.Adam(),\
                  loss='sparse_categorical_crossentropy', \
                  metrics=['accuracy'])

    model.summary()

    history = model.fit(train_x, train_y, epochs=30, batch_size=256, validation_split=0.25)

    # visualization
    plt.figure(figsize=(12, 4))

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
    print('loss: {}, accuracy: {}'.format(*model.evaluate(test_x, test_y, verbose=0)))


if __name__ == "__main__":
    cnn_fashion_mnist_exec()
