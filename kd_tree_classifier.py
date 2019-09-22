"""
Reza Marzban
https://github.com/Reza-Marzban
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from assignmet1.simple_dataset_creator import SimpleDataSetCreator
from assignmet1.complex_dataset_creator import ComplexDataSetCreator


class KdTreeClassifier:
    tree = None
    training_x = None
    training_y = None

    def fit(self, x, y):
        """
        :param x: training data features
        :param y: training data labels
        """
        self.training_x = x
        self.training_y = y
        print(" ")
        self.tree = cKDTree(x, 1)
        print(f"Fitted the kd_tree nearest neighbor classifier model on {len(x)} datapoints.")

    def predict_and_evaluate(self, x, y):
        """
        :param x: testing data features
        :param y: testing data labels
        return: predicted values, and the prediction accuracy
        """
        _, i = self.tree.query(x, k=1)
        y_hat = self.training_y[i]
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

        plt.title(f"KD Tree Classifier -- accuracy: {accuracy}%")
        plt.legend()
        plt.show()
        print(" ")
        return y_hat, accuracy


if __name__ == "__main__":
    x_train, y_train, x_test, y_test = SimpleDataSetCreator.create_simple_dataset()
    kd_tree_classifier = KdTreeClassifier()
    kd_tree_classifier.fit(x_train, y_train)
    kd_tree_classifier.predict_and_evaluate(x_test, y_test)
    x_train, y_train, x_test, y_test = ComplexDataSetCreator.create_complex_dataset()
    kd_tree_classifier = KdTreeClassifier()
    kd_tree_classifier.fit(x_train, y_train)
    kd_tree_classifier.predict_and_evaluate(x_test, y_test)
    print()
