"""
Reza Marzban
https://github.com/Reza-Marzban
"""
import numpy as np


class KDTree:
    def __init__(self, matrix):
        """
        Constructor of KD tree that works with 2-D data and find the 1 nearest neighbour. (k=1, d=2)
        :param matrix: an n*d numpy array with n data points and d features (d should be 2)
        """
        self.n, self.d = matrix.shape
        self.kd_tree = self._create_tree_recursively(matrix)
        self.candidate = None

    def find_nearest(self, vector):
        self._find_closest_leaf(vector)
        nearest_neighbor = self.candidate
        self.candidate = None
        return nearest_neighbor

    def _create_tree_recursively(self, matrix, step=0):
        # base condition (create a leaf)
        if len(matrix) == 1:
            tree = Node(data=matrix[0])
            return tree

        relevant_d = step % self.d
        sorted_matrix = matrix[matrix[:, relevant_d].argsort()]
        median = len(sorted_matrix) // 2
        left_data = sorted_matrix[:median]
        if len(left_data) == 0:
            left_data = sorted_matrix[median]
            right_data = sorted_matrix[median+1:]
        else:
            right_data = sorted_matrix[median:]
        tree = Node(data=sorted_matrix[median][relevant_d],
                    left_child=self._create_tree_recursively(left_data, step+1),
                    right_child=self._create_tree_recursively(right_data, step+1))
        return tree

    def _find_closest_leaf(self, vector, tree=None, step=0):
        if tree is None:
            tree = self.kd_tree

        if tree.left is None and tree.right is None:
            if self.candidate is None:
                self.candidate = tree.data
            elif self._calculate_distance(vector, self.candidate) > self._calculate_distance(vector, tree.data):
                self.candidate = tree.data

        else:
            relevant_d = step % self.d
            if vector[relevant_d] >= tree.data:
                self._find_closest_leaf(vector, tree.right, step+1)
                if (vector[relevant_d]-tree.data) <= (vector[relevant_d]-self.candidate[relevant_d]):
                    self._find_closest_leaf(vector, tree.left, step+1)
            else:
                self._find_closest_leaf(vector, tree.left, step+1)
                if (vector[relevant_d]-tree.data) <= (vector[relevant_d]-self.candidate[relevant_d]):
                    self._find_closest_leaf(vector, tree.right, step+1)

    def _calculate_distance(self, vector1, vector2):
        euclidean_distance = np.sqrt((vector1[0] - vector2[0]) ** 2 + (vector1[1] - vector2[1]) ** 2)
        return euclidean_distance


class Node:
    def __init__(self, data=None, left_child=None, right_child=None):
        """
        if the left_child and right_child are none, then it is a leaf and data is the actual datapoint.
        else it is not a leaf and data is just the median seprator of relevant_d
        """
        self.left = left_child
        self.right = right_child
        self.data = data


if __name__ == "__main__":
    print("")
    print("Testing our KDtree algorithm")
    print("data:")
    x, y = np.mgrid[0.:5., 2.:8.]
    data = np.c_[x.ravel(), y.ravel()]
    print(data)
    kd_tree = KDTree(data)
    vector = np.array([2.1, 2.9])
    print(f"Vector: {[2.1, 2.9]}")
    nn1 = kd_tree.find_nearest(vector)
    print("printing the nearest neighbour by my algorithm:")
    print(nn1)

    from scipy.spatial import cKDTree
    tree = cKDTree(data, 1)
    _, i = tree.query([2.1, 2.9], k=1)
    nn2 = tree.data[i]
    print("printing the nearest neighbour by scipy.spatial.cKDTree:")
    print(nn2)
    print("")



