# Stage 9: Customer Segmentation — Data Science Interview Guide

This guide provides technical and conceptual answers to key data science interview questions covering **Unsupervised Learning**, **K-Means Clustering**, **Feature Scaling**, and **Cluster Evaluation**.

---

### Q1: What is Unsupervised Machine Learning?
**Answer:**  
Unsupervised machine learning is a class of algorithms that analyzes unlabelled datasets to discover underlying patterns, groupings, or structural relationships without explicit target supervision. Unlike supervised learning (which predicts a target $y$ given features $X$), unsupervised learning works exclusively with features $X$. Examples include clustering (K-Means, Hierarchical), dimensionality reduction (PCA, t-SNE), and anomaly detection.

---

### Q2: What is Clustering?
**Answer:**  
Clustering is an unsupervised learning task that partitions a dataset into groups (clusters) such that data points within the same cluster are mathematically more similar to each other (high intra-cluster similarity) than to points in other clusters (low inter-cluster similarity).

---

### Q3: How does the K-Means Clustering algorithm work step-by-step?
**Answer:**  
K-Means is an iterative distance-based partition algorithm that works in 4 steps:
1. **Initialization**: Select $K$ initial cluster centroids randomly (or using K-Means++ initialization to spread centroids far apart).
2. **Assignment Step**: Assign each data point $x_i$ to its nearest centroid based on Euclidean distance ($d(x, c) = \sqrt{\sum (x_j - c_j)^2}$).
3. **Update Step**: Recalculate each centroid as the component-wise mean (average) of all data points assigned to that cluster.
4. **Convergence Check**: Repeat steps 2 and 3 until centroids stop moving (convergence) or until the maximum number of iterations is reached.

---

### Q4: What is a Cluster Centroid?
**Answer:**  
A cluster centroid is the mathematical center of a cluster in multi-dimensional feature space. It represents a vector of feature means across all observations assigned to that cluster. In K-Means, the centroid acts as the representative archetype of that specific segment.

---

### Q5: Why is Feature Scaling (e.g. StandardScaler) mandatory before K-Means clustering?
**Answer:**  
K-Means calculates distances using Euclidean distance. Features measured on larger numerical scales (e.g. `TotalCharges` ranging from $0 to $8,600) will dominate the distance calculation over features with smaller numerical scales (e.g. `ServiceCount` ranging from 0 to 8). `StandardScaler` standardizes each feature to zero mean ($\mu = 0$) and unit variance ($\sigma = 1$), ensuring all features contribute equally to geometric distances.

---

### Q6: What is the Elbow Method and how does it help select the optimal number of clusters (K)?
**Answer:**  
The Elbow Method plots the Within-Cluster Sum of Squares (Inertia) against varying values of $K$. As $K$ increases, inertia strictly decreases because clusters become smaller and closer to data points. The "elbow point" is the point of diminishing returns where adding another cluster yields significantly smaller reductions in inertia. This elbow indicates an optimal balance between compactness and model simplicity.

---

### Q7: What is the Silhouette Score and how is it interpreted?
**Answer:**  
The Silhouette Score measures how well-separated clusters are on a scale from $-1$ to $+1$:
$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$
where $a(i)$ is the mean intra-cluster distance for point $i$, and $b(i)$ is the mean distance from point $i$ to the nearest neighboring cluster.
- **$+1$**: Data point is far away from neighboring clusters (excellent cluster separation).
- **$0$**: Data point lies right on the boundary between two clusters.
- **$-1$**: Data point may have been assigned to the wrong cluster.

---

### Q8: Why was the target variable `Churn` strictly excluded during cluster formation?
**Answer:**  
Including `Churn` would introduce **supervised leakage** into an unsupervised task. The goal of segmentation is to discover natural, unbiassed customer behavioral groups based solely on financial and engagement metrics (`tenure`, `MonthlyCharges`, `TotalCharges`, `ServiceCount`). Excluding `Churn` preserves clustering integrity. `Churn` was analyzed **post-hoc** (after clustering) to measure the empirical churn rate of each discovered segment.

---

### Q9: What is the fundamental difference between Supervised Classification and Unsupervised Clustering?
**Answer:**  
- **Supervised Classification** uses labeled training data $(X, y)$ to learn a mapping function $f(X) \to y$ to predict discrete target classes (e.g., predicting whether a customer will Churn: Yes/No).
- **Unsupervised Clustering** uses unlabeled data $(X)$ to partition samples into natural subsets based on geometric proximity, without prior knowledge of group labels.

---

### Q10: Does belonging to a high-churn cluster mean the segment causes churn?
**Answer:**  
No. Clustering identifies **correlation and co-occurrence**, not **causation**. Observing a 46.4% churn rate in the "High-Spend At-Risk Onboarders" segment means customers with high monthly charges and short tenure exhibit higher historical churn probability. It does not mean being in that segment forces a customer to leave.

---

### Q11: What are the main limitations of K-Means Clustering?
**Answer:**  
1. **Sensitivity to Initial Centroids**: Poor initial centroids can lead to suboptimal local minima (mitigated by using `n_init > 1` and K-Means++).
2. **Assumes Spherical Clusters**: K-Means assumes clusters are convex and isotropic (spherical); it struggles with complex non-linear shapes.
3. **Sensitivity to Outliers**: Extreme outliers pull centroids away from true cluster centers.
4. **Pre-specified K**: The algorithm requires the user to specify $K$ manually prior to execution.
