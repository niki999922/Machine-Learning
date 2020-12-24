import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import adjusted_rand_score

FIELD_NUM_WITH_CLASS = 7
EPS = 1e-6
LIMIT_ITERATIONS = 1000


def read_dataset(name):
    return pd.read_csv(name)


def paint_points(y_arr, x_features_with_norm_2d, title="title"):
    colours = ["r", "m", "b", "k", "c", "y", "g"]
    plt.figure(figsize=(8, 8))
    unique_labels = np.unique(y_arr)
    for i in range(len(unique_labels)):
        label = unique_labels[i]
        cur_x_arr = x_features_with_norm_2d[y_arr == label, 0]
        cur_y_arr = x_features_with_norm_2d[y_arr == label, 1]
        plt.scatter(cur_x_arr, cur_y_arr, color=colours[i], alpha=0.5, label=label)
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.show()


class KMeans:
    def __init__(self, clusters):
        self.clusters = clusters
        self.centroids = list()

    def init_centroids(self, x_arr):
        rows, columns = x_arr.shape
        centroids = list()
        centroids.append(x_arr[np.random.randint(0, rows)])
        self.centroids = centroids
        for _ in range(self.clusters - 1):
            min_x_distances_to_centroids = list()
            for x in x_arr:
                min_x_distances_to_centroids.append(self.find_nearest_centroid(x)[1])
            sorted_min_x_distances_to_centroids = np.argsort(min_x_distances_to_centroids)
            sum_min_distances = np.sum(min_x_distances_to_centroids)
            p = random.random()
            while p == 0:
                p = random.random()
            cur_dist = p * sum_min_distances
            cur_id = 0
            cur_sum = min_x_distances_to_centroids[sorted_min_x_distances_to_centroids[cur_id]]
            while cur_sum < cur_dist:
                cur_id += 1
                cur_sum += min_x_distances_to_centroids[sorted_min_x_distances_to_centroids[cur_id]]
            centroids.append(x_arr[sorted_min_x_distances_to_centroids[cur_id]])

    def find_nearest_centroid(self, x):
        distances = list()
        for c in self.centroids:
            distances.append(np.linalg.norm(c - x))
        idx = np.argmin(distances)
        return idx, distances[idx]

    def teach(self, x_arr):
        self.init_centroids(x_arr)
        rows, columns = x_arr.shape
        for _ in range(LIMIT_ITERATIONS):
            clusters_size, clusters_x_sum = np.zeros(self.clusters), np.zeros((self.clusters, columns))
            for x in x_arr:
                c_id = self.find_nearest_centroid(x)[0]
                clusters_size[c_id] += 1
                clusters_x_sum[c_id] += x
            new_centroids = np.copy(self.centroids)
            for i in range(self.clusters):
                if clusters_size[i] != 0:
                    new_centroids[i] = clusters_x_sum[i] / clusters_size[i]
            diff = list()
            for x in (self.centroids - new_centroids):
                diff.append(np.linalg.norm(x))
            if np.max(diff) <= EPS:
                break
            self.centroids = new_centroids

    def predict(self, x_arr):
        predicted = list()
        for x in x_arr:
            predicted.append(self.find_nearest_centroid(x)[0])
        return predicted


def inner_cluster_cohesion(x_features_with_norm, y_predicted):
    result = 0
    unique_labels = np.unique(y_predicted)
    for i in range(len(unique_labels)):
        label = unique_labels[i]
        indexes = np.where(y_predicted == label)
        x_arr_of_label = x_features_with_norm[indexes]
        rows = x_arr_of_label.shape[0]
        if rows == 0:
            continue
        centroid = np.sum(x_arr_of_label, axis=0) / rows
        distances = list()
        for x in x_arr_of_label:
            distances.append(np.linalg.norm(x - centroid) ** 2)
        sum_dist = np.sum(distances)
        result += sum_dist / rows
    return result


def inner_calinski_harabasz(x_features_with_norm, y_predicted, clusters):
    N = len(y_predicted)
    K = clusters
    unique_labels = np.unique(y_predicted)
    top, bottom = 0, 0
    centroids = list()
    for i in range(len(unique_labels)):
        label = unique_labels[i]
        indexes = np.where(y_predicted == label)
        x_arr_of_label = x_features_with_norm[indexes]
        rows = x_arr_of_label.shape[0]
        if rows == 0:
            continue
        centroid = np.sum(x_arr_of_label, axis=0) / rows
        centroids.append(centroid)
        distances = list()
        for x in x_arr_of_label:
            distances.append(np.linalg.norm(x - centroid))
        sum_dist = np.sum(distances)
        bottom += sum_dist

    centroid_main = np.sum(centroids, axis=0) / len(centroids)

    for i in range(len(unique_labels)):
        indexes = np.where(y_predicted == unique_labels[i])
        M = len(indexes[0])
        diff = np.linalg.norm(centroids[i] - centroid_main)
        top += M * diff
    result = ((N - K) * top) / ((K - 1) * bottom)
    return result


def paint_metric(clusters, metrics, title="title"):
    plt.figure(figsize=(14, 9))
    plt.grid(linestyle='-')
    plt.plot(clusters, metrics, linestyle='-', marker='.', color='g')
    plt.title(title)
    plt.xlabel("Clusters")
    plt.show()


def pictures_distribution(x_features_with_norm, x_features_with_norm_2d, y_real):
    k_means = KMeans(clusters=3)
    k_means.teach(x_features_with_norm)
    y_predicted = k_means.predict(x_features_with_norm)
    paint_points(y_real, x_features_with_norm_2d, title="Real dataset")
    paint_points(y_predicted, x_features_with_norm_2d, title="KMeans")


def graph_cluster_score(x_features_with_norm, y_real, max_cluster_size=10):
    x_arr, inner_metric_y_arr, external_metric_y_arr, = [], [], []
    for clusters in range(2, max_cluster_size):
        x_arr.append(clusters)
        k_means_tmp = KMeans(clusters=clusters)
        k_means_tmp.teach(x_features_with_norm)
        y_predicted_tmp = k_means_tmp.predict(x_features_with_norm)
        inner_metric_y_arr.append(inner_calinski_harabasz(x_features_with_norm, y_predicted_tmp, clusters=clusters))
        external_metric_y_arr.append(adjusted_rand_score(y_real, y_predicted_tmp))

    paint_metric(x_arr, inner_metric_y_arr, "Calinski–Harabasz")
    paint_metric(x_arr, external_metric_y_arr, "Adjusted rand score")


if __name__ == '__main__':
    print("Starting")
    data = read_dataset('dataset.csv')
    reducer = PCA(n_components=2)

    y_real = np.array(list(map(lambda el: el - 1, data.loc[:, data.columns[FIELD_NUM_WITH_CLASS]])))
    x_features = data[data.columns[:FIELD_NUM_WITH_CLASS]]  # because in end of dataset classes
    x_features_with_norm = MinMaxScaler(copy=True, feature_range=(0, 1)).fit_transform(x_features)
    x_features_with_norm_2d = reducer.fit_transform(x_features_with_norm)

    pictures_distribution(x_features_with_norm, x_features_with_norm_2d, y_real)
    graph_cluster_score(x_features_with_norm, y_real, max_cluster_size=12)
    print("End")
