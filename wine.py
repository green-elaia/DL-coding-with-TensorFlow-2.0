import pandas as pd
import matplotlib.pyplot as plt



def wine_exec():
    load_data()
    preprocess_data()
    run_train()
    run_test()


def load_data():
    red = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv', \
                      sep=';')
    white = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv', \
                        sep=';')
    # print(red.head(), '\n', white.head())
    red['type'] = 0
    white['type'] = 1
    # print(red.head(2), '\n', white.head(2))
    wine = pd.concat([red, white])
    # print(wine.describe())
    # plt.hist(wine['type'])
    # plt.xticks([0, 1])
    # plt.show()
    # print(wine['type'].value_counts())


if __name__ == '__main__':
    load_data()