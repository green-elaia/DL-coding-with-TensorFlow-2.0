import tensorflow as tf
import matplotlib.pyplot as plt


def exec():
    load_data()
    run_train()
    run_test()



def load_data():
    global train_x, train_y, test_x, test_y
    boston_housing_data = tf.keras.datasets.boston_housing
    (train_x, train_y), (test_x, test_y) = boston_housing_data.load_data()

    # standardization
    # 각 feature의 단위가 다르므로 이것을 정규화하여 학습효율을 높이도록 한다.
    # 정규화 할 때는 test set의 정보를 제외하도록 하여 모델의 일반화 성능을 왜곡시키지 않도록 한다.
    x_mean = train_x.mean(axis=0)
    x_std = train_x.std(axis=0)
    y_mean = train_y.mean(axis=0)
    y_std = train_y.std(axis=0)

    train_x = (train_x - x_mean) / x_std
    train_y = (train_y - y_mean) / y_std
    test_x = (test_x - x_mean) / x_std
    test_y = (test_y - y_mean) / y_std



def run_train():
    global train_x, train_y, model
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(units=52, activation='relu', input_shape=(len(train_x[0]),)))
    model.add(tf.keras.layers.Dense(units=39, activation='relu'))
    model.add(tf.keras.layers.Dense(units=26, activation='relu'))
    model.add(tf.keras.layers.Dense(units=1))

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.07), loss='mse')
    model.summary()

    history = model.fit(train_x, train_y, batch_size=32, epochs=25, validation_split=0.25)

    # visualization
    plt.plot(history.history['loss'], 'b-', label='loss')
    plt.plot(history.history['val_loss'], 'r--', label='val_loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()



def run_test():
    global test_x, test_y, model
    model.evaluate(test_x, test_y)

    # visualization
    pred_y = model.predict(test_x)
    plt.figure(figsize=(5,5))
    plt.plot(test_y, pred_y, 'b.')
    plt.axis([min(test_y), max(test_y), min(test_y), max(test_y)])
    plt.plot([min(test_y), max(test_y)], [min(test_y), max(test_y)], ls='--', c='.3')
    plt.xlabel('test_y')
    plt.ylabel('pred_y')
    plt.show()




























