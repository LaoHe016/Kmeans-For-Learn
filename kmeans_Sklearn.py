from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import numpy as np


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

# 生成数据
X = 带状数据()

# 训练模型
model = KMeans(n_clusters=2, init='k-means++', n_init=1)
model.fit(X)

# 可视化结果
plt.scatter(X[:, 0], X[:, 1], c=model.labels_, cmap='viridis', s=50)
plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], 
            c='red', marker='X', s=200, label='Centroids')
plt.title("Scikit-learn 实现结果")
plt.legend()
plt.show()