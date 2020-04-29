### Basic Transformer

Last update: April 2020.

---

Implementations of a few attention-based [1] models in PyTorch. We follow the notation of the Set Transformer [2], since much of our codebase was built on theirs. Of course, the Annotated Transformer [3] was helpful as well.

**Attention**

For set of queries $Q$, keys $K$, and values $V$, we define:
$$
\text{Attention}(Q,K,V) = \text{Softmax}\left(\frac{1}{\sqrt{d_k}}QK^\top\right)V.
$$


The interpretation is that each query $q_k$ corresponds to a different feature, which is constructed as a convex combination of values $v_k$, where the weights are given by the dot products between the query and the keys $k_k$. To see this, we can expand:
$$
\text{Softmax}\left(\frac{1}{\sqrt d}
\begin{bmatrix}
q_1^\top k_1 & \dots & q_1^\top k_n\\
\vdots & & \vdots \\
q_n^\top k_1 & \dots & q_n^\top k_n
\end{bmatrix}\right) \begin{bmatrix}-\ v_1\ -\\\vdots\\-\ v_n\ -\end{bmatrix} = 
\begin{bmatrix}-\ \sum_{i=1}^n w_i^{(1)}v_i -\\ \vdots \\ -\ \sum_{i=1}^n w_i^{(n)}v_n\ -\end{bmatrix},
$$


where the softmax is applied row-by-row, corresponding to a softmax per query. [Note: In the above $w_i^{(j)}$ corresponds to the weights over $v_1,\dots,v_n$ that arise from softmax due to the $j$-th query vector $q_j$.]

We normalize by $\frac{1}{\sqrt d_k}$ (the dimensionality of the queries and keys) to prevent the dot products $q_i^\top k_j$ from blowing up in value and thereby vanishing gradients of the softmax. [Note: Recall that if $q \sim N(0, I)$ and $k\sim N(0,I)$ then $q^\top k\sim N(0, d_k)$.]

**Multi-head attention block (MAB)**

The multi-head attention block takes queries $Q$ and values $V$, and treats the values $V$ as keys $K$. It uses $h$ heads to project the queries, keys, and values into $h$ different embeddings through linear transforms. Then it applies attention to each of the $h$ embeddings, concatenates the resulting features at the end, then adds it to the result of another feed-forward neural network at the end like a residual block.
$$
\begin{align*}
\mathrm{MAB}(Q,V) & =[O_1,\dots,O_h] + \mathrm{ReLU}([O_1,\dots,O_h]W^O)\\
O_i & = \text{Attention}(QW_i^Q, VW_i^K,VW_i^V)
\end{align*}
$$


Note that above $\{W_i^Q,W_i^K,W_i^V\}_{i=1}^h$ and $W^O$ are trainable parameters. 

**Set attention block (SAB)**

This is just a MAB with self-attention over elements of the set.
$$
\mathrm{SAB}(X) = \mathrm{MAB}(X,X)
$$


**Induced set attention block (ISAB)**

To reduce the computational complexity of self-attention we replace the values with a set of $m$ induced values. These values are determined by attending over $X$ with a set of $m$ inducing points $I$ as query vectors. 
$$
\begin{align*}
	\mathrm{ISAB}(X) &= \mathrm{MAB}(X,H),\\
	H & = \mathrm{MAB}(I, X)
\end{align*}
$$


**Pooling by multi-head attention (PMA)**

After applying layers of set attention blocks, we want to pool the output into a set of $k$ elements. To do so we learn a set of $k$ seed vectors $S$ and which are queried over the elements $X$.
$$
\mathrm{PMA}_k(X) = \mathrm{MAB}(S,X)
$$

#### References

[1] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, Ł., and Polosukhin, I. (2017). Attention is All you Need. In Advances in Neural Information Processing Systems 30, I. Guyon, U.V. Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, eds. (Curran Associates, Inc.), pp. 5998–6008.

[2] Lee, J., Lee, Y., Kim, J., Kosiorek, A., Choi, S., and Teh, Y.W. (2019). Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks. In International Conference on Machine Learning, pp. 3744–3753.

[3] G. Klein, Y. Kim, Y. Deng, J. Senellart, & A. Rush, OpenNMT: Open-Source Toolkit for Neural Machine Translation. Proceedings of ACL 2017, System Demonstrations (Vancouver, Canada: Association for Computational Linguistics, 2017), pp. 67–72.