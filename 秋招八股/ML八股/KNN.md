##### 1.KNN优缺点：
- 优点：
	- 简单直观：kNN 算法简单易懂，容易实现和理解。
	- 无需训练：kNN 是一种基于实例的学习方法，不需要进行显式的训练过程，只需要存储训练样本即可。
	- 对异常值不敏感：kNN 算法对于数据中的异常值不敏感，因为预测时是基于邻居样本的投票或平均值来决定分类或回归结果。
- 缺点：
	- 计算复杂度高：kNN 算法需要计算样本之间的距离，随着样本数量增加，计算复杂度也会增加，尤其是在高维数据集上。
	- 内存消耗大：kNN 需要存储训练样本的特征向量，对于大规模数据集，会占用较大的内存空间
	- 需要确定 k 值：kNN 算法需要事先确定 k 值，选择不合适的 k 值可能会影响预测结果。
	- 数据不平衡问题：如果训练数据中某个类别的样本数远远多于其他类别，kNN 算法在预测时可能会有偏差。
##### 2.KNN需要进行归一化处理特征吗：
- 需要
	- 距离敏感性： kNN算法通过计算数据点之间的距离来找出最近的k个邻居，进而进行分类或回归。如果数据集中的特征在量纲上差异很大，未归一化的数据可能会导致某些特征在距离计算中占主导地位，从而影响算法的准确性。
	- 防止数据溢出： 在进行距离计算时，特别是使用欧氏距离计算方法时，如果数据的数值范围非常大，可能会导致计算过程中出现数值溢出的问题。归一化通过将数据缩放到一个较小的范围来避免这一问题。
##### 3.KNN的K设置过大过小的问题
- 过小：
	- 模型容易受到噪声和异常点的影响，因为少数几个噪声点就可能对预测结果产生较大的影响。
	- 模型过于复杂，容易过拟合训练数据，导致泛化能力较差。
- 过大：
	- 欠拟合（Underfitting）： 当k值过大，模型可能太过简化，导致无法捕捉数据的复杂性和模式，从而在训练和测试数据上都表现不佳
	- 对数据不平衡问题敏感： 如果数据集中某些类别的样本数量远多于其他类别，过大的k值可能导致少数类别的样本在预测时被多数类别所“淹没”，从而降低少数类别样本的预测准确性。
##### 4.kNN 算法在处理高维数据时有什么问题？
- 问题：
	- 维度灾难（Curse of Dimensionality）：随着数据维度的增加，数据空间呈指数增长。这意味着在高维空间中，数据变得稀疏，样本之间的距离变得更加不可靠，导致 kNN 算法的性能下降。
	- 距离计算困难：在高维空间中，距离计算变得更加复杂和耗时。高维数据中的距离计算涉及大量特征，可能导致计算开销大大增加。
	- 特征的相关性和冗余： 在高维数据中，特征之间可能存在大量的相关性或冗余信息。这不仅增加了处理数据的计算负担，而且还可能降低kNN等算法的性能，因为不是所有特征都对决策过程有用
- 如何解决：
	- 特征选择或者降维
	- 距离度量： 使用更适合高维数据的距离度量方法，如余弦相似度等。
##### 5.如何处理不平衡数据集的情况下的 kNN 分类问题？
- 重采样技术：通过欠采样或过采样来平衡数据集中不同类别的样本数量。欠采样可以随机减少多数类样本，而过采样可以通过复制或合成少数类样本来增加其数量。
- 距离权重：在kNN算法中引入距离权重机制，使得距离近的样本对分类结果的贡献更大。可以使用距离的倒数或其他衡量样本间相似性的指标来计算权重。
- 阈值调整：根据具体问题的需求，调整分类的决策阈值。可以将决策阈值设置为适应不平衡数据集的值，使得分类器更倾向于较少类别的预测。
##### 6.kNN 算法如何处理缺失值？
- 删除包含缺失值的样本：最简单的方法是删除包含缺失值的样本。这种方法适用于数据集中缺失值较少的情况，且确保删除样本不会导致信息的严重丢失。
- 缺失值填充：使用合适的方法填充缺失值。常见的填充方法包括均值填充、中位数填充、众数填充等。对于数值型特征，可以使用属性的均值、中位数或其他统计量来填充缺失值。对于分类特征，可以使用众数进行填充。
- 特征加权：在计算距离时，可以考虑对特征进行加权处理，以减少缺失值对距离计算的影响。对于缺失特征，可以给予较低的权重，以降低其对距离计算的贡献。
##### 7.手撕KNN
```
import numpy as np

def distance(x1, x2):
	return np.sqrt(np.sum((x1 - x2) ** 2))

class KNN:
	def __init__(self, k=3):
		self.k = k

	def fit(self, X, y):
		self.X = X
		self.y = y

	def predict(self, X):
		y_pred = [self._predict(x) for x in X]
		return y_pred

	def _predict(self, x):
		distances = [distance(x, x_train) for x_train in self.X]
		k_indices = np.argsort(distances)[:self.k]
		k_labels = [self.y[i] for i in k_indices]
		y = np.bincount(k_labels).argmax()
		return y


## 并行计算
def knn_no_loops(data, query, k):
    """
    Perform a k-nearest neighbor search without using explicit loops.

    Args:
    data (numpy.ndarray): The dataset to search against, where each row is a data point.
    query (numpy.ndarray): The query point, as a 1D numpy array.
    k (int): The number of nearest neighbors to return.

    Returns:
    numpy.ndarray: The indices of the k nearest neighbors.
    """
    # 计算差值
    diff = data - query
    # 计算欧氏距离的平方
    dist_squared = np.sum(diff ** 2, axis=1)
    # 获取最小的k个距离的索引
    nearest_neighbors = np.argsort(dist_squared)[:k]
    return nearest_neighbors
```
##### 7.如何加速
- **KD 树（k-dimensional tree, KD-Tree）**：
    - KD 树是一种用于组织 k 维空间中点的二叉树结构，特别适用于低维数据。
    - 构建 KD 树的时间复杂度为 O(nlog⁡n)，查找最近邻的时间复杂度平均为 O(log⁡n)。
    - 适用于维度较低的数据集（通常维度在20以下），对于高维数据效果较差。
- **球树（Ball Tree）**：
    - 球树是一种将数据点组织在多维空间中的分层树结构，适用于高维数据。
    - 树节点表示空间中的一个超球体，每个节点的子节点代表该超球体的进一步划分。
    - 构建和查询的效率在高维数据中优于 KD 树。
- **最近邻搜索的近似算法（Approximate Nearest Neighbor Search）**：
    - **局部敏感哈希（Locality-Sensitive Hashing, LSH）**：将数据点投影到低维空间中，通过哈希函数将相似的数据点映射到相同的桶中。
- **聚类方法**：
    - **K-means 聚类**：先对数据进行 K-means 聚类，然后在每个簇内进行 KNN 搜索。查询时，首先找到查询点所属的簇，然后在该簇内进行 KNN 搜索。
- **降维技术**：
    - **主成分分析（PCA）**：通过线性变换将高维数据投影到低维空间中，以减少数据维度，提高查询效率。
    - **t-SNE（t-distributed Stochastic Neighbor Embedding）**：用于将高维数据嵌入到低维空间中，特别适用于可视化。