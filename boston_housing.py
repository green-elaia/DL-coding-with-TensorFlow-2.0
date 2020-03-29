import tensorflow as tf
import matplotlib.pyplot as plt


def boston_housing_exec():
    load_data()
    preprocess_data()
    run_train()
    run_test()



def load_data():
    global train_x, train_y, test_x, test_y
    boston_housing_data = tf.keras.datasets.boston_housing
    (train_x, train_y), (test_x, test_y) = boston_housing_data.load_data()



def preprocess_data():
    """
    standardization (z-score normalization)
    각 feature의 단위가 다르므로 이것을 정규화하여 학습효율을 높이도록 한다.
    이것은 학습과정에서 0으로 수렴하거나 발산하지 않도록 해준다.
    정규화 할 때는 test set의 정보를 반영하지 않도록 train set으로만 평균과 표준편차를 계산한다.
    이것은 모델의 일반화 성능 왜곡을 방지한다.
    """
    global train_x, train_y, test_x, test_y
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

    # callback 인수는 매 epoch이 끝날때마다 지정한 리스트에 명시한 함수를 호출함
    # Earlystopping function의 경우 monitor에서 지정한 것을 대상으로 하여
    # 지정한 patience 만큼의 epoch 동안 최고기록을 갱신하지 못하면 학습을 멈추게 함
    # 이것은 overfitting을 막을 수 있음
    history = model.fit(train_x, train_y, batch_size=32, epochs=25, validation_split=0.25, \
                        callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, monitor='val_loss')])

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
    # option c는 점선의 grayscale 정도를 명시. 0.0~1.0 사이의 값이고 0에 가까울수록 검은색
    plt.plot([min(test_y), max(test_y)], [min(test_y), max(test_y)], ls='--', c='.3')
    plt.xlabel('test_y')
    plt.ylabel('pred_y')
    plt.show()



if __name__ == "__main__":
    boston_housing_exec()
