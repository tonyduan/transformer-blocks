### Transformer Models

Last update: April 2023.

---

Implementations of blocks for attention-based models:

1. Encoder and decoder blocks of the original Transformer paper (Vaswani et al. 2017).
2. Cross-attention block from the Perceiver IO paper (Jaegle et al. 2022).
3. Induced point block from the Set Transformer paper (Lee et al. 2019).
4. Linear attention variants of the above blocks (Katharopoulos et al. 2020).

We implement a pre-normalization scheme with ScaleNorm throughout (Nguyen and Salazar 2019).

#### Scaled Dot-Product Attention

For set of queries $\mathbf{Q} \in\mathbb{R}^{m,d}$, keys $\mathbf{K}\in\mathbb{R}^{n,d}$, and values $\mathbf{V}\in \mathbb{R}^{n,d}$, we define:

```math
\text{Attention}(\mathbf{Q},\mathbf{K},\mathbf{V}) = \text{Softmax}\left(\frac{1}{\sqrt{d}}\mathbf{Q}\mathbf{K}^\top\right)\mathbf{V}.
```

Note each matrix consists of *row vectors*, not column vectors.

The interpretation is that each query $\mathbf{q}_j$ corresponds to a different feature which aggregates a convex combination of the values $\mathbf{v}_i$, where the weights of the convex combination are given by a scaled softmax over the dot products between query $\mathbf{q}_j$ and the keys $\mathbf{k}_i$. To see this, we can expand

```math
\text{Softmax}\left(\frac{1}{\sqrt d}
\begin{bmatrix}
\mathbf{q}_1^\top \mathbf{k}_1 & \dots & \mathbf{q}_1^\top \mathbf{k}_n\\
\vdots & & \vdots \\
\mathbf{q}_n^\top \mathbf{k}_1 & \dots & \mathbf{q}_n^\top \mathbf{k}_n
\end{bmatrix}\right) \begin{bmatrix}-\ \mathbf{v}_1\ -\\\vdots\\-\ \mathbf{v}_n\ -\end{bmatrix} =
\begin{bmatrix}-\ \sum_i w_i^{(1)}\mathbf{v}_i -\\ \vdots \\ -\ \sum_i w_i^{(n)}\mathbf{v}_i\ -\end{bmatrix},
```

where the softmax is applied row-by-row, corresponding to a softmax per query. Here, the weights corresponding to the query $\mathbf{q}_j$ comprise an $n$-dimensional vector
```math
\mathbf{w}^{(j)} = \mathrm{Softmax}\left(\frac{1}{\sqrt d} \begin{bmatrix} \mathbf{q}_j^\top \mathbf{k}_1 & \dots &\mathbf{q}_j^\top \mathbf{k}_n\end{bmatrix}\right),\quad\quad \mathbf{w}^{(j)} \geq \mathbf{0},\quad\quad\mathbf{1}^\top \mathbf{w^{(j)}}=1
```

We normalize by $\frac{1}{\sqrt d_k}$ (the dimensionality of the queries and keys) to prevent the dot products $q_i^\top k_j$ from blowing up in value and thereby vanishing gradients in the softmax.

(Recall that if $\mathbf{q} \sim N(\mathbf{0}, \mathbf{I})$ and $\mathbf{k}\sim N(\mathbf{0},\mathbf{I})$ then $\mathbf{q}^\top \mathbf{k}\sim N(0, d)$.)

Runtime is $O(mnd)$ i.e. "quadratic" in the self-attention case where $m=n$.

#### Multi-Head Attention

Multi-head attention projects $\mathbf{Q},\mathbf{K},\mathbf{V}$ using $h$ heads through linear embeddings. Then it applies attention to each of the embeddings, concatenates the resulting features at the end, then sends the result through another linear transform to create outputs $\mathbf{O}$.

```math
\begin{align*}
\mathrm{MultiHeadAttention}(\mathbf{Q},\mathbf{K},\mathbf{V}) & = \begin{bmatrix}\mid & & \mid\\
\mathbf{O}_1&\dots & \mathbf{O}_h\\
\mid & & \mid
\end{bmatrix}\mathbf{W}^{(\mathbf{O})}\\
\mathrm{where}\ \mathbf{O}_i & = \text{Attention}(\mathbf{Q}\mathbf{W}_i^\mathbf{Q}, \mathbf{K}\mathbf{W}_i^\mathbf{K},\mathbf{V}\mathbf{W}_i^\mathbf{V})\\
\mathrm{and}\ & (\mathbf{W}_i^\mathbf{Q},\mathbf{W}_i^\mathbf{K},\mathbf{W}_i^\mathbf{V})_{i=1}^h, \mathbf{W}^\mathbf{O}\ \text{are learnable parameters}
\end{align*}
```
Note that this operation is permutation-equivariant with respect to the inputs $\mathbf{Q}$, and $\mathbf{K}$, $\mathbf{V}$.

#### Linear Attention

Consider the output of attention corresponding to the query $\mathbf{q}_j$. Earlier we saw that it is equal to a convex combination of the values $\mathbf{v}_i$, with weights given using keys $\mathbf{k}_i$. We can generalize the computation of these weights by expressing it as a kernel function $\mathrm{sim}(\cdot, \cdot)$ between queries and keys. Of course, we'll have to normalize so that the weights still sum to one. Scaled dot-product attention is equivalent to the following.
```math
\sum_i w_i^{(j)}\mathbf{v}_i = \frac{\sum_i \mathrm{sim}(\mathbf{q}_j,\mathbf{k}_i) \mathbf{v}_i}{\sum_i\mathrm{sim}(\mathbf{q}_j,\mathbf{k}_i)}, \quad\quad \mathrm{sim}(\mathbf{q}_j,\mathbf{k}_i) = \exp\left(\frac{1}{\sqrt d}\mathbf{q}_j^\top \mathbf{k}_i\right).
```

Recall that we can express any kernel function as an inner product in a feature space, say  $p$-dimensional. Furthermore recall the associative property of matrix multiplication. We can write the following.
```math
\begin{align*}
\sum_i w_i^{(j)}\mathbf{v}_i & = \frac{\sum_i \phi(\mathbf{q}_j)^\top \phi(\mathbf{k}_i) \mathbf{v}_i}{\sum_i\phi(\mathbf{q}_j)^\top \phi(\mathbf{k}_i)}\\
& = \frac{\phi(\mathbf{q}_j)^\top \sum_i\phi(\mathbf{k}_i) \mathbf{v}_i^\top}{\phi(\mathbf{q}_j)^\top \sum_i \phi(\mathbf{k}_i)}\\
\end{align*}
```
More succinctly, we can write:
```math
\mathrm{LinearAttention}(\mathbf{Q},\mathbf{K},\mathbf{V}) = \frac{\phi(\mathbf{Q})(\phi(\mathbf{K})^\top \mathbf{V})}{\phi(\mathbf{Q})(\phi(\mathbf{K})^\top\mathbf{1})}
```
Observe that both $\phi(\mathbf{K})^\top \mathbf{V}$ and $\phi(\mathbf{K})^\top \mathbf{1}$ only need to be computed once. The runtime is $O((m+n)pd)$. This is extremely favorable because typically $m \approx n \gg p \approx d$, and this brings runtime down to linear.

The authors propose the following feature representation in which $p=d$.
```math
\phi(\mathbf{x}) = \mathrm{ELU}(\mathbf{x})+1
```
Concurrent work Shen et al. 2021 proposes the feature representation
```math
\phi(\mathbf{x}) = \exp(\mathbf{x}),
```
which approximates the original kernel function via
```math
\mathrm{sim}(\mathbf{q}_j,\mathbf{k}_j) = \exp(\mathbf{q}_j)^\top \exp(\mathbf{k}_i) \approx \exp\left(\frac{1}{\sqrt d}\mathbf{q}_j^\top \mathbf{k}_i\right).
```

We can interpret linear attention as an RNN in the case of a causal attention mask (typically used in a Transformer decoder). Then each query $\mathbf{q}_j$ can only attend to keys $\mathbf{k}_i$ where $i \leq j$. That is, $\mathrm{sim}(\mathbf{q}_j, \mathbf{k}_i) = 0$for $i >j$.  We can interpret $\sum_i \phi(\mathbf{k}_i)\mathbf{v}_i$ and $\phi(\mathbf{k}_i)$ as a "hidden state" that is updated in a recurrence relation for each subsequent query $\mathbf{q}_j$.

#### Language Model

In this repository we implement a simple language model using the WikiText-2 dataset.

For this task given a corpus we try to learn the distribution
```math
p_\theta(\mathbf{x}^{(i)}|\mathbf{x}^{(i-1)},\dots,\mathbf{x}^{(i-L)}) \sim \mathrm{Categorical}(\boldsymbol\pi),
```
using a set of Transformer encoder blocks followed by an attention-based pooling layer.

Note that the English vocabulary consists of roughly 30k tokens in size.

#### References

[1] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, Ł., and Polosukhin, I. (2017). Attention is All you Need. In Advances in Neural Information Processing Systems 30, I. Guyon, U.V. Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, eds. (Curran Associates, Inc.), pp. 5998–6008.

[2] Lee, J., Lee, Y., Kim, J., Kosiorek, A., Choi, S., and Teh, Y.W. (2019). Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks. In International Conference on Machine Learning, pp. 3744–3753.

[3] Nguyen, T. Q. & Salazar, J. Transformers without Tears: Improving the Normalization of Self-Attention. in Proceedings of the 16th International Conference on Spoken Language Translation (Association for Computational Linguistics, 2019).

[4] Katharopoulos, A., Vyas, A., Pappas, N. & Fleuret, F. Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention. in *International Conference on Machine Learning* 5156–5165 (PMLR, 2020).

[5] Shen, Z., Zhang, M., Zhao, H., Yi, S. & Li, H. Efficient Attention: Attention With Linear Complexities. in *2021 Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision* 3531–3539 (2021).
