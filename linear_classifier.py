"""
Reza Marzban
https://github.com/Reza-Marzban
"""
import numpy as np
import matplotlib.pyplot as plt
from assignmet1.simple_dataset_creator import SimpleDataSetCreator
from assignmet1.complex_dataset_creator import ComplexDataSetCreator


class LinearClassifier:
    B = None
    training_x = None
    training_y = None

    def fit(self, x, y):
        """
        :param x: training data features
        :param y: training data labels
        :return: weights
        """
        self.training_x = x
        self.training_y = y
        print(" ")
        x_transpose = np.transpose(x)
        temp = np.dot(np.linalg.inv(np.dot(x_transpose, x)), x_transpose)
        self.B = np.dot(temp, y)
        print(f"Fitted the Linear classifier model on {len(x)} datapoints.")
        return self.B

    def predict_and_evaluate(self, x, y):
        """
        :param x: testing data features
        :param y: testing data labels
        return: predicted values, and the prediction accuracy
        """
        regression = np.dot(x, self.B)
        y_hat = (regression >= 0.5) * 1
        accuracy = round(((y == y_hat).sum()) / len(y) * 100, 2)
        print(f"Accuracy of {len(x)} test datapoints: {accuracy}%")

        train_0_mask = np.where(self.training_y.squeeze() == 0)
        training_0_features = self.training_x[train_0_mask]
        train_1_mask = np.where(self.training_y.squeeze() == 1)
        training_1_features = self.training_x[train_1_mask]

        test_tp_mask = np.logical_and(y == 1, y_hat == 1).squeeze()
        test_tn_mask = np.logical_and(y == 0, y_hat == 0).squeeze()
        test_fn_mask = np.logical_and(y == 1, y_hat == 0).squeeze()
        test_fp_mask = np.logical_and(y == 0, y_hat == 1).squeeze()

        test_tp = x[test_tp_mask]
        test_tn = x[test_tn_mask]
        test_fn = x[test_fn_mask]
        test_fp = x[test_fp_mask]

        alpha = 0.34
        alpha2 = 0.66
        size = 30

        plt.scatter(training_0_features[:, 0], training_0_features[:, 1],
                    s=size, alpha=alpha, label='Training class 0')
        plt.scatter(training_1_features[:, 0], training_1_features[:, 1],
                    s=size, alpha=alpha, label='Training class 1')
        plt.scatter(test_fn[:, 0], test_fn[:, 1],
                    s=size, alpha=alpha2, label='False Negative')
        plt.scatter(test_fp[:, 0], test_fp[:, 1],
                    s=size, alpha=alpha2, label='False Positive')
        plt.scatter(test_tp[:, 0], test_tp[:, 1],
                    s=size, alpha=alpha, label='True Positive')
        plt.scatter(test_tn[:, 0], test_tn[:, 1],
                    s=size, alpha=alpha, label='True Negative')

        plt.title(f"Linear Classifier -- accuracy: {accuracy}%")
        plt.legend()
        plt.show()
        print(" ")
        return y_hat, accuracy


if __name__ == "__main__":
    x_train, y_train, x_test, y_test = SimpleDataSetCreator.create_simple_dataset()
    lc = LinearClassifier()
    lc.fit(x_train, y_train)
    lc.predict_and_evaluate(x_test, y_test)
    x_train, y_train, x_test, y_test = ComplexDataSetCreator.create_complex_dataset()
    lc = LinearClassifier()
    lc.fit(x_train, y_train)
    lc.predict_and_evaluate(x_test, y_test)
    print()
