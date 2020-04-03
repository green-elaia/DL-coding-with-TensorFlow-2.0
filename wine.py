import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf


def wine_exec():
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
    # print(red.head(), '\n', white.head())
    red['type'] = 0
    white['type'] = 1
    # print(red.head(2), '\n', white.head(2))
    wine = pd.concat([red, white])
    # print(wine.describe())   # dataframe의 간단한 통계정보 확인
    # plt.hist(wine['type'])
    # plt.xticks([0, 1])
    # plt.show()
    # print(wine['type'].value_counts())



def preprocess_data():
    """
    min-max normalization
    shuffle the data
    convert to numpy array
    separate the data into train and test dataset
    """
    global wine, train_x, train_y, test_x, test_y
    # print(wine.info())   # dataframe의 attiribute의 정보 요약
    wine_norm = (wine - wine.min(axis=0)) / (wine.max(axis=0) - wine.min(axis=0))
    # print(wine_norm.head(), '\n', wine_norm.describe())
    wine_shuffle = wine_norm.sample(frac=1)
    # print(wine_shuffle.head())
    wine_np = wine_shuffle.to_numpy()
    # print(wine_np[:5])

    test_start_idx = int(len(wine_np) * 0.8)
    train_x, train_y = wine_np[:test_start_idx, :-1], wine_np[:test_start_idx, -1]
    test_x, test_y = wine_np[test_start_idx:, :-1], wine_np[test_start_idx:, -1]
    # print(train_x[0], train_y[0], '\n', test_x[0], test_y[0])

    # convert to on-hot encoding
    train_y = tf.keras.utils.to_categorical(train_y, num_classes=2)
    test_y = tf.keras.utils.to_categorical(test_y, num_classes=2)
    # print(train_y[0], '\n', test_y[0])



def run_train():
    global train_x, train_y, model
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(units=48, activation='relu', input_shape=(12,)))
    model.add(tf.keras.layers.Dense(units=24, activation='relu'))
    model.add(tf.keras.layers.Dense(units=12, activation='relu'))
    model.add(tf.keras.layers.Dense(units=2, activation='softmax'))

    """
    loss function의 사용과 관련된 참고사항
    
    There are three kinds of classification tasks:
    1. Binary classification: two exclusive classes
    2. Multi-class classification: more than two exclusive classes
    3. Multi-label classification: just non-exclusive classes
    
    Here, we can say
    In the case of (1), you need to use binary cross entropy.
    In the case of (2), you need to use categorical cross entropy.
    In the case of (3), you need to use binary cross entropy.

    You can just consider the multi-label classifier as a combination of multiple independent binary classifiers.
    If you have 10 classes here, you have 10 binary classifiers separately.
    Each binary classifier is trained independently.
    Thus, we can produce multi-label for each sample.
    If you want to make sure at least one label must be acquired,
    then you can select the one with the lowest classification loss function, or using other metrics.

    I want to emphasize that multi-class classification is not similar to multi-label classification!
    Rather, multi-label classifier borrows an idea from the binary classifier!
    """
    """
    categorical cross entropy(CCE)  vs  sparse categorical cross entropy
    
    CCE와 sparse CCE는 수학적으로는 동일한 loss function.
    차이점은,
    정답이 one-hot encoding 방식으로 되어 있으면 'CCE'를 쓰고
    정답이 integer 방식으로 되어 있으면 'sparse CCE'를 쓴다.
    """
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01), \
                  loss='categorical_crossentropy', \
                  metrics=['accuracy'])
    model.summary()

    history = model.fit(train_x, train_y, epochs=30, batch_size=64, validation_split=0.25, \
                        callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, monitor='val_loss')])

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
    wine_exec()
