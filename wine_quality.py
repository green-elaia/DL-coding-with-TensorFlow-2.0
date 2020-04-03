import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


def wine_quality_exec():
    load_data()
    preprocess_data()
    run_train()
    run_test()



def load_data():
    global wine
    red = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv', \
                      sep=';')
    white = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv', \
                        sep=';')
    wine = pd.concat([red, white])
    # print(wine['quality'].describe(), '\n', wine['quality'].value_counts())
    # plt.hist(wine['quality'], bins=7, rwidth=0.8)
    # plt.show()

    # value 0 : quality 3~5, 품질 나쁨
    # value 1 : quality 6, 품질 보통
    # value 2 : quality 6~9, 품질 좋음
    wine.loc[wine['quality'] <= 5, 'new_quality'] = 0
    wine.loc[wine['quality'] == 6, 'new_quality'] = 1
    wine.loc[wine['quality'] >= 7, 'new_quality'] = 2
    # print(wine['new_quality'].describe(), '\n', wine['new_quality'].value_counts())



def preprocess_data():
    """
    min-max normalization
    shuffle the data
    convert to numpy array
    separate the data into train and test dataset
    """
    global wine, train_x, train_y, test_x, test_y
    del wine['quality']
    wine_norm = (wine - wine.min(axis=0)) / (wine.max(axis=0) - wine.min(axis=0))
    wine_shuffle = wine_norm.sample(frac=1)
    wine_np = wine_shuffle.to_numpy()

    test_start_idx = int(len(wine_np) * 0.8)
    train_x, train_y = wine_np[:test_start_idx, :-1], wine_np[:test_start_idx, -1]
    test_x, test_y = wine_np[test_start_idx:, :-1], wine_np[test_start_idx:, -1]

    # 정답 one-hot encoding
    train_y = tf.keras.utils.to_categorical(train_y, num_classes=3)
    test_y = tf.keras.utils.to_categorical(test_y, num_classes=3)



def run_train():
    global train_x, train_y, model
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(units=48, activation='relu', input_shape=(train_x.shape[1],)))
    model.add(tf.keras.layers.Dense(units=24, activation='relu'))
    model.add(tf.keras.layers.Dense(units=12, activation='relu'))
    model.add(tf.keras.layers.Dense(units=3, activation='softmax'))

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01), loss='categorical_crossentropy', \
                  metrics=['accuracy'])

    model.summary()

    history = model.fit(train_x, train_y, epochs=30, batch_size=64, validation_split=0.25, \
                        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4)])

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
    plt.xlabel('Epoch')
    plt.ylim(0.7, 1)
    plt.legend()

    plt.show()



def run_test():
    global test_x, test_y, model
    model.evaluate(test_x, test_y)


if __name__ == '__main__':
    wine_quality_exec()
