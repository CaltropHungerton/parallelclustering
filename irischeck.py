import csv
import numpy as np
from sklearn.cluster import KMeans

def read_csv(file_path):
    data = []
    with open(file_path, 'r') as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            data.append([float(x) for x in row])
    return np.array(data)

file_path = 'iris_edit.csv'
data = read_csv(file_path)
n_clusters = 3

kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(data)

print(f"Centroids dimensions: {kmeans.cluster_centers_.shape}")

print("Centroids:", kmeans.cluster_centers_)
