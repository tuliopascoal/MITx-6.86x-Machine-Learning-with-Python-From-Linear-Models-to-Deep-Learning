from sklearn_extra.cluster import KMedoids
import numpy as np

X = np.asarray([[0, -6], [4, 4], [0, 0], [-5, 2]])
kmedoids1 = KMedoids(n_clusters=2, metric='euclidean', random_state=0).fit(X)
#kmedoids2 = KMedoids(n_clusters=2, metric="manhattan", max_iter=2).fit(X)
kmedoids2 = KMedoids(n_clusters=2, metric="manhattan", random_state=0).fit(X)

print("kmedoids1.labels_:", kmedoids1.labels_)
print("kmedoids1.predict([[0,-6], [4,4], [0, 0], [-5,2]]): ", kmedoids1.predict([[0,-6], [4,4], [0, 0], [-5,2]]))
print("kmedoids1.cluster_centers_:", kmedoids1.cluster_centers_)
print("kmedoids1.medoid_indices_:", kmedoids1.medoid_indices_)
print("kmedoids1.inertia_:", kmedoids1.inertia_)
print("\n\n")

print("kmedoids2.labels_:", kmedoids2.labels_)
print("kmedoids2.predict([[0,-6], [4,4], [0, 0], [-5,2]]): ", kmedoids2.predict([[0,-6], [4,4], [0, 0], [-5,2]]))
print("kmedoids2.cluster_centers_:", kmedoids2.cluster_centers_)
print("kmedoids2.medoid_indices_:", kmedoids2.medoid_indices_)
print("kmedoids2.inertia_:", kmedoids2.inertia_)