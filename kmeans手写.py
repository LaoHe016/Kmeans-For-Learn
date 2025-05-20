import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

def 均匀数据():
    return np.random.rand(300, 2)

def 双月牙数据():
    from sklearn.datasets import make_moons

    # 生成数据
    n_samples = 500  # 样本数量
    noise = 0.05     # 噪声比例（控制数据分散程度）
    random_state = 42 # 随机种子（确保可重复性）

    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)

    # 可视化
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=30, edgecolors='k')
    plt.title("Scikit-learn 生成的月牙形数据（含两个簇）")
    plt.xlabel("特征1")
    plt.ylabel("特征2")
    plt.show()
    return X

def 同心圆数据():
    from sklearn.datasets import make_circles
    X_circle, _ = make_circles(n_samples=300, noise=0.05, factor=0.5)
    plt.scatter(X_circle[:, 0], X_circle[:, 1], s=50)
    plt.show()
    return X_circle

def 带状数据():
    X_band, _ = make_blobs(n_samples=300, centers=3, cluster_std=[1.0, 0.5, 0.3], random_state=42)
    plt.scatter(X_band[:, 0], X_band[:, 1], s=50)
    plt.show()
    return X_band

class KMeans:
    def __init__(self, k=3, max_iter=100):
        self.k = k
        self.max_iter = max_iter
    
    def fit(self, X):
        # 随机初始化中心点（基于数据范围）
        self.centroids = X[np.random.choice(range(len(X)), self.k, replace=False)]
        
        for _ in range(self.max_iter):
            # 分配样本到最近的中心点
            labels = self._assign_clusters(X)
            # 可视化当前步骤
            self._plot_step(X, labels, f"Iteration {_+1}")
            # 更新中心点
            new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.k)])
            # 若中心点无变化则停止
            if np.allclose(self.centroids, new_centroids):
                break
            self.centroids = new_centroids
    
    def _assign_clusters(self, X):
        distances = np.sqrt(((X[:, np.newaxis] - self.centroids)**2).sum(axis=2))
        return np.argmin(distances, axis=1)
    
    def _plot_step(self, X, labels, title):
        plt.figure(figsize=(8, 6))
        plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50)
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], c='red', s=200, marker='X')
        plt.title(title)
        plt.show()

# 生成测试数据（均匀分布）
X = 带状数据()
kmeans = KMeans(k=3)
kmeans.fit(X)