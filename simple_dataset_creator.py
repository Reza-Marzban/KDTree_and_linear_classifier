"""
Reza Marzban
https://github.com/Reza-Marzban
"""
import numpy as np
import matplotlib.pyplot as plt


class SimpleDataSetCreator:

    @staticmethod
    def create_simple_dataset():
        """Create 10000 datapoint with two classes"""
        # first dataset class distribution
        mean = [2, 1]
        cov = [[1, 0], [0, 3]]
        data1 = np.random.multivariate_normal(mean, cov, 5000)
        # Second dataset class distribution
        mean = [5, 8]
        cov = [[3, 1], [1, 3]]
        data2 = np.random.multivariate_normal(mean, cov, 5000)

        x = np.concatenate((data1, data2))

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
    x_train, y_train, x_test, y_test = SimpleDataSetCreator.create_simple_dataset()

    plt.scatter(x_train[:, 0], x_train[:, 1], s=30, alpha=0.32, c=y_train.squeeze(), cmap="Paired")
    plt.show()

    y_test1 = y_test+5

    plt.scatter(np.concatenate((x_train[:, 0], x_test[:, 0])), np.concatenate((x_train[:, 1], x_test[:, 1])),
                s=30, alpha=0.3, c=np.concatenate((y_train,y_test1)).squeeze(), cmap="Paired")
    plt.show()

