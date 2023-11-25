## Settings
There are $n$ devices and they are partitioned into K groups. A client $i$'s model can be written as $x_i=[x_{i,s},x_{i,p}]$. For devices within the same group k, there are some layers parameters that can be shared ($x_{i,s}$) and some are private ($x_{i,p}$). The parameters of all models among all devices are denoted as $X=[x_1,\ldots,x_n]$.

<!-- Suppose each device has a loss function $f_i(x_i)$. Denote $\hat{x}_{k,s}$ as the shared parameters of the groundtruth group
$$
\hat{x}_{k,s} = \frac{1}{|G_k|} \sum_{i\in G_k} x_{i,s}
$$
and therefore $\hat{x}_i=[\hat{x}_{k,s}; x_{i,p}]$. Note that:
- These notations are used only for analysis, but not in the algorithm.
- If the grouping is perfect, then $x_{i,s}=\hat{x}_{k,s}$ for all $i\in G_k$.

Then we can define the **group objective** as
$$ F_k(X; G_k) = \frac{1}{|G_k|} \sum_{i \in G_k} f_i\left(\hat{x}_{i}\right). $$
Note that clients belong to one group if their shared layers satisfy the "generalized strong growth condition".

We would like that eventually, each group $k$ has 
$$
\|\nabla F_k(X; G_k)\|_2 \leq \epsilon.
$$
This is the objective we want to show in the end. -->


## The Optimization Objectives
Let us define a matrix space for the graph
$$
\mathcal{S} = \{ W: W\in\{0,1\}^{n\times n}, W=W^\top, W_{ii}=1 \forall~i\in[n] \}
$$
- $W_{ij}=1$ if $i$ and $j$ are connected.
- Due to "generalized strong growth condition" is transitional, all clients in the same group are connected. Without loss of generality, we may assume the matrices in $\mathcal{S}$ are block diagonal.

### If K is known
Let us define a matrix space for the graph with rank K
$$
\mathcal{S}_K = S \cap \{ W: \text{rank}(W) = K \}.
$$
The optimization objective becomes
$$ 
\min_{W\in\mathcal{S}_K} F(W), \qquad \text{where } F(W):=\max_{\lambda_{ij}\in\mathbb{R}^{d} } \min_{X\in\mathbb{R}^{d\times n}} \left[ \frac{1}{n} \sum_{i=1}^n f_i(x_i) + \sum_{i<j} w_{ij}\left<\lambda_{ij}, x_{i,s} - x_{j,s} \right>
\right].
$$
Note that the second term is a hard constraint that $x_{i,s} = x_{j,s}$ if $w_{ij}=1$, but in the algorithm where we may relax this a bit. 

**Algorithms that optimize this objective**: A draft of the algorithm can be devised as follows:
- Optimizing $\lambda_{ij}$ means we synch shared layers within a group. (still use GD)
- Optimizing $X$ using SGD as usual.
- Optimizing $W$. We can consider an easier case first, e.g. $\min_{W\in \{0,1\}^{n(n-1)/2} } F(W)$.
    - Approach 1: Submodular Optimization (the diminishing returns property may not hold)
        - (**how can we view this as a submodular optimization problem**) Let's assume $\{f_i\}$ are convex and for non-convex $f_i$ we use $\|\nabla f_i(x_i)\|_2^2$ inside $F$ instead. Let's imagine in the very beginning, the $W$ is an identity matrix, then $F(W)$ is the sum of $n$ objectives that optimized independently. The loss is the smallest. When we add an edge to the graph (say $i$, $j$), then $F(W)$ becomes evaluating $[(x_{i,s}+x_{j,s})/2, x_{i,p}]$ on $f_i$ and $[(x_{i,s}+x_{j,s})/2, x_{j,p}]$ on $f_j$  and evaluate $f_k(x_k)$ otherwise. (e.g., $f_i(x_i^\star)+f_j(x_j^\star)\le f_i(x^\star)+f_j(x^\star)$)
            - If $i$ and $j$ belong to the same groundtruth group (meaning they can simultaneously reach minimizer or stationary point), then the loss do not increase*. 
            - If $i$ and $j$ belong to two different groundtruth groups, then the loss will increase a lot. 

            In other words, adding edges will make the loss non-decrease. ~~It may also satisfy the diminishing returns: suppose there are two groundtruth groups with 1 group of size n-1 identical nodes and 1 group of size 1 (say node A), then adding node A to a smaller group will increase loss more than adding A to a larger group.  (TODO: need rigorous proof)~~
        - **Not submodular nor supermodular.** We only have super-additive.
        - (TODO: how is this related to the Matroid Rank Functions?)
        - (**algorithm?**). 
    - Approach 2: Spectral Partitioning of the graph.
        - See the description below. (we can use the gradient of shared layers as the representation of each node)
        - TODO: (Is it easy to analyze?)
    - Approach 3: Relax the $W$ space to continuous and add a penalty term to make it sparse.
        - then we can use gradient descent to optimize it.
        - TODO: https://arxiv.org/pdf/1311.4296.pdf (uses frank-wolfe)
    - Approach 4: Relax and solve using Frank-Wolfe algorithm
        - If we let $w_{ij}\in[0,1]$ instead of $\{0,1\}$ and $\{f_i\}$ are convex, (but now the matrix space is non-convex)
            -  we can use Frank-Wolfe algorithm to optimize it.


### If K is unknown

We add a penalty term to the objective $ - \alpha 1^\top W 1$ which encourages the number of groups to be small. The optimization objective becomes
$$ 
\min_{W\in\mathcal{S}} F(W) - \alpha 1^\top W 1,
$$
The $\alpha$ is a hyperparameter that we can tune. One possibility is to look at the norm of gradient of shared layers. If alpha is very large, then it encourages less but larger groups. Then the gradient of shared layers will be small will not converge to 0. In this case we should lower the alpha.

 
## GPT description of spectral clustering
The spectral method for the graph partition problem is a technique that uses the properties of the eigenvalues and eigenvectors of matrices associated with a graph, such as the adjacency matrix or the Laplacian matrix, to find an optimal or near-optimal partition of the graph. This method is particularly useful for identifying clusters or communities within the graph.

### Key Concepts:

1. **Adjacency Matrix**: For a graph \( G \) with \( n \) vertices, the adjacency matrix \( A \) is an \( n \times n \) matrix where \( A_{ij} \) is 1 if there is an edge between vertices \( i \) and \( j \), and 0 otherwise.

2. **Laplacian Matrix**: The Laplacian matrix \( L \) of a graph is defined as \( L = D - A \), where \( D \) is the diagonal matrix of vertex degrees and \( A \) is the adjacency matrix. The Laplacian matrix has several important properties useful in graph partitioning.

### Spectral Partitioning Steps:

1. **Compute the Laplacian Matrix**: Calculate \( L = D - A \).

2. **Eigenvalue Decomposition**: Find the eigenvalues and eigenvectors of the Laplacian matrix. The eigenvalues are non-negative and can be sorted in ascending order: \( 0 = \lambda_1 \leq \lambda_2 \leq \ldots \leq \lambda_n \).

3. **Fiedler Vector**: The eigenvector corresponding to the second smallest eigenvalue (known as the Fiedler value) is used for partitioning the graph. This eigenvector is known as the Fiedler vector.

4. **Partition the Graph**: Use the sign of the components of the Fiedler vector to partition the vertices into two groups. Typically, vertices corresponding to positive components are put in one group and those with negative components are put in another.

5. **Recursive Partitioning**: For k-way partitioning, this process can be recursively applied to the resulting subgraphs.

### Intuition:

- The Fiedler vector tends to have a property where vertices connected by high-weight edges or in dense parts of the graph are likely to have similar signs. This makes it effective for identifying clusters or communities within the graph.

- Minimizing the edge cut between partitions is closely related to the second smallest eigenvalue of the Laplacian (the Fiedler value), as per the Cheeger's inequality in spectral graph theory.

### Applications and Limitations:

- **Applications**: Spectral partitioning is widely used in areas like data clustering, image segmentation, and network analysis.

- **Limitations**: The method can have difficulties with graphs having uneven cluster sizes or very sparse connectivity. It also tends to work best for graphs where a good partition is closely related to the graph's spectral properties.

In summary, the spectral method is a powerful technique for the graph partition problem, leveraging the algebraic properties of the graph's Laplacian matrix to find meaningful partitions based on the graph's inherent structure.


## GPT description of Use Spectral Partitioning for Data Clustering
Spectral partitioning can be effectively used for data clustering by transforming the data into a graph representation and then applying spectral methods to partition this graph. The process involves several steps, starting from representing the data as a graph and ending with the application of clustering algorithms based on the spectral properties of the graph.

### Steps to Use Spectral Partitioning for Data Clustering:

1. **Data Representation as a Graph**:
   - **Vertices**: Each data point is represented as a vertex in the graph.
   - **Edges**: Edges are created between vertices, often based on the similarity or distance between the corresponding data points. The choice of how to define edges and edge weights depends on the nature of the data and the clustering objectives.

2. **Construction of the Similarity Matrix**:
   - This matrix (often an adjacency matrix) represents the similarities or distances between data points. For instance, in case of Euclidean distance, the entry \( A_{ij} \) might be \( \exp(-\|x_i - x_j\|^2 / 2\sigma^2) \), where \( x_i \) and \( x_j \) are data points and \( \sigma \) is a scaling parameter.

3. **Formation of the Laplacian Matrix**:
   - The Laplacian matrix \( L \) is computed from the similarity matrix. The most commonly used form is the normalized Laplacian, defined as \( L = I - D^{-1/2}AD^{-1/2} \), where \( D \) is the diagonal degree matrix and \( A \) is the adjacency matrix.

4. **Eigenvalue Decomposition**:
   - Compute the eigenvalues and eigenvectors of the Laplacian matrix. For clustering, we are particularly interested in the eigenvectors corresponding to the smallest eigenvalues.

5. **Use Eigenvectors for Clustering**:
   - Form a feature vector for each data point from the eigenvectors. For instance, if using the first \( k \) eigenvectors, the feature vector for the \( i \)-th data point could be formed from the \( i \)-th components of these eigenvectors.
   - This step effectively maps the original data points into a new space where clusters are more distinguishable.

6. **Apply a Clustering Algorithm**:
   - Use a standard clustering algorithm like k-means on these feature vectors to identify clusters. The transformed space often makes it easier for these algorithms to detect clusters, even if the original space had complex structures.

7. **Interpretation of Clusters**:
   - The resulting clusters from the k-means (or other algorithms) in the transformed space correspond to clusters in the original data.

### Advantages and Applications:

- **Handling Complex Structures**: Spectral clustering is particularly good at identifying clusters with non-convex shapes and can handle clusters of different sizes and densities better than some traditional methods like k-means.
- **Applications**: Widely used in image segmentation, social network analysis, and any field where data can be naturally represented as a graph.

### Considerations:

- **Parameter Selection**: The choice of parameters like the number of clusters \( k \) and the scaling parameter \( \sigma \) can significantly affect the results.
- **Scalability**: Spectral clustering can be computationally intensive, especially for very large datasets.

In essence, spectral partitioning for data clustering leverages the spectral properties of graphs derived from data to identify natural clusters, making it a powerful tool for uncovering the underlying structure in complex datasets.