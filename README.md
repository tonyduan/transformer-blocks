### Transformer Models

Last update: December 2022.

---

Implementations of blocks for attention-based models; the encoder/decoder blocks of the original Transformer [1] and the permutation-equivariant blocks of the Set Transformer [2]. We implement pre-normalization via ScaleNorm as described in [3].

**Scaled Dot-Product Attention**

For set of queries $Q$, keys $K$, and values $V$, we define:

```math
\text{Attention}(Q,K,V) = \text{Softmax}\left(\frac{1}{\sqrt{d_k}}QK^\top\right)V.
```

The interpretation is that each query $q_k$ corresponds to a different feature, which is constructed as a convex combination of values $v_k$, where the weights are given by the dot products between the query and the keys $k_k$. To see this, we can expand (note each matrix consists of *row vectors*, not column vectors):
```math
\text{Softmax}\left(\frac{1}{\sqrt d}
\begin{bmatrix}
q_1^\top k_1 & \dots & q_1^\top k_n\\
\vdots & & \vdots \\
q_n^\top k_1 & \dots & q_n^\top k_n
\end{bmatrix}\right) \begin{bmatrix}-\ v_1\ -\\\vdots\\-\ v_n\ -\end{bmatrix} =
\begin{bmatrix}-\ \sum_i w_i^{(1)}v_i -\\ \vdots \\ -\ \sum_i w_i^{(n)}v_n\ -\end{bmatrix},
```

where the softmax is applied row-by-row, corresponding to a softmax per query. (Note: In the above $w_i^{(j)}$ corresponds to the weights over $v_1,\dots,v_n$ that arise from softmax due to the $j$-th query vector $q_j$.)

We normalize by $\frac{1}{\sqrt d_k}$ (the dimensionality of the queries and keys) to prevent the dot products $q_i^\top k_j$ from blowing up in value and thereby vanishing gradients of the softmax. (Note: Recall that if $q \sim N(0, I)$ and $k\sim N(0,I)$ then $q^\top k\sim N(0, d_k)$.)

**Multi-Head Attention**

Multi-head attention projects $Q,V,K$ using $h$ heads through linear transforms. Then it applies attention to each of the $h$ embeddings, concatenates the resulting features at the end, then sends the result through another linear transform.

```math
\begin{align*}
\mathrm{MultiHeadAttn}(Q,V) & = \mathrm{Concat}(O_1,\dots,O_h)W^O\\
\mathrm{where}\ O_i & = \text{Attention}(QW_i^Q, VW_i^K,VW_i^V)
\end{align*}
```

Above $\{W_i^Q,W_i^K,W_i^V\}_{i=1}^h$ and $W^O$ are trainable parameters. Note that the operation is permutation-equivariant.

**Language Model**

In this repository we implement a simple language model using the WikiText-2 dataset. For this task given a corpus we try to learn the distribution

```math
p(x^{(i)}\ |\ x^{(i-L)},\dots,x^{(i-1)}) \sim \mathrm{Categorical}
```
using a set of Transformer encoder blocks followed by an attention-based pooling layer. Note that the English vocabulary has roughly 30k tokens in size.

#### References

[1] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, Ł., and Polosukhin, I. (2017). Attention is All you Need. In Advances in Neural Information Processing Systems 30, I. Guyon, U.V. Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, eds. (Curran Associates, Inc.), pp. 5998–6008.

[2] Lee, J., Lee, Y., Kim, J., Kosiorek, A., Choi, S., and Teh, Y.W. (2019). Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks. In International Conference on Machine Learning, pp. 3744–3753.

[3] Nguyen, T. Q. & Salazar, J. Transformers without Tears: Improving the Normalization of Self-Attention. in Proceedings of the 16th International Conference on Spoken Language Translation (Association for Computational Linguistics, 2019).
