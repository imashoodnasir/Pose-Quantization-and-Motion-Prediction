from sklearn.cluster import KMeans
import torch

def kmeans_quantization(poses, num_clusters=10):
    kmeans = KMeans(n_clusters=num_clusters)
    labels = kmeans.fit_predict(poses.reshape(-1, poses.shape[-1]))
    return labels

if __name__ == "__main__":
    dummy_poses = torch.randn(100, 17, 3).numpy()
    labels = kmeans_quantization(dummy_poses)
    print("K-Means Clusters:", labels)
