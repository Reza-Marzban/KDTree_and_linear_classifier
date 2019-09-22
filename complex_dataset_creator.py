"""
Reza Marzban
https://github.com/Reza-Marzban
"""
import numpy as np
import matplotlib.pyplot as plt


class ComplexDataSetCreator:

    @staticmethod
    def create_complex_dataset():
        """Create 10000 datapoint with two classes"""
        # first dataset class distribution
        mean = [1, 1]
        cov = [[1, 0], [0, 0.5]]
        data1 = np.random.multivariate_normal(mean, cov, 1000)
        mean = [1, 8]
        cov = [[1, 1], [0, 2]]
        data2 = np.random.multivariate_normal(mean, cov, 1000)
        mean = [7, 4]
        cov = [[0, 1], [1.5, 0.5]]
        data3 = np.random.multivariate_normal(mean, cov, 1000)
        mean = [0, 7]
        cov = [[0.5, 0], [0, 1]]
        data4 = np.random.multivariate_normal(mean, cov, 1000)
        mean = [3, 10]
        cov = [[0.5, 1], [1, 0]]
        data5 = np.random.multivariate_normal(mean, cov, 1000)

        # Second dataset class distribution
        mean = [5, 8]
        cov = [[2, 1], [1, 2]]
        data6 = np.random.multivariate_normal(mean, cov, 1000)
        mean = [3, 3]
        cov = [[0, 1], [1, 0]]
        data7 = np.random.multivariate_normal(mean, cov, 1000)
        mean = [2, 8]
        cov = [[0.2, 0], [1, 1]]
        data8 = np.random.multivariate_normal(mean, cov, 1000)
        mean = [1, 4]
        cov = [[1, 0.5], [2, 0]]
        data9 = np.random.multivariate_normal(mean, cov, 1000)
        mean = [6, 1]
        cov = [[1, 0], [0, 0.5]]
        data10 = np.random.multivariate_normal(mean, cov, 1000)

        x = np.concatenate((data1, data2, data3, data4, data5, data6, data7, data8, data9, data10))

        # creating the labels
        y1 = np.zeros((5000, 1))
        y2 = np.ones((5000, 1))
        y = np.concatenate((y1, y2))

        # splitting to train and test set (0.85, 0.20)
        mask = np.random.rand(10000) < 0.85
        x_train = x[mask]
        y_train = y[mask]
        mask = np.logical_not(mask)
        x_test = x[mask]
        y_test = y[mask]

        return x_train, y_train, x_test, y_test


if __name__ == "__main__":
    x_train, y_train, x_test, y_test = ComplexDataSetCreator.create_complex_dataset()

    plt.scatter(x_train[:, 0], x_train[:, 1], s=30, alpha=0.32, c=y_train.squeeze(), cmap="Paired")
    plt.show()

    y_test1 = y_test+5

    plt.scatter(np.concatenate((x_train[:, 0], x_test[:, 0])), np.concatenate((x_train[:, 1], x_test[:, 1])),
                s=30, alpha=0.3, c=np.concatenate((y_train,y_test1)).squeeze(), cmap="Paired")
    plt.show()

