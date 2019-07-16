### Transformer Models 

Last update: June 2019.

---

Implementations of a few attention-based models in PyTorch.

We follow the notation of the Set Transformer [1], since much of our codebase was built on theirs.

**Attention**

For set of queries <img alt="$Q$" src="svgs/1afcdb0f704394b16fe85fb40c45ca7a.svg" align="middle" width="12.99542474999999pt" height="22.465723500000017pt"/>, keys <img alt="$K$" src="svgs/d6328eaebbcd5c358f426dbea4bdbf70.svg" align="middle" width="15.13700594999999pt" height="22.465723500000017pt"/>, and values <img alt="$V$" src="svgs/a9a3a4a202d80326bda413b5562d5cd1.svg" align="middle" width="13.242037049999992pt" height="22.465723500000017pt"/>, we define [2]:
<p align="center"><img alt="$$&#10;\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{1}{\sqrt{d}}QK^\top\right)V.&#10;$$" src="svgs/433869ff025e5d7aeb0b2e7b3d4d8f57.svg" align="middle" width="325.6154517pt" height="39.452455349999994pt"/></p>


The interpretation is that each query <img alt="$q_k$" src="svgs/acc433299a99053a1abda8186a07965d.svg" align="middle" width="14.604341399999988pt" height="14.15524440000002pt"/> corresponds to a different feature, which is constructed as a convex combination of values <img alt="$v_k$" src="svgs/eaf0887cdc4cb5f8e69a7796f143c3eb.svg" align="middle" width="15.23409524999999pt" height="14.15524440000002pt"/>, where the weights are given by the dot products between the query and the keys <img alt="$k_k$" src="svgs/1271cba53526e0c04aa67fe6b06300be.svg" align="middle" width="15.82390589999999pt" height="22.831056599999986pt"/>. To see this, we can expand:
<p align="center"><img alt="$$&#10;\text{softmax}\left(\frac{1}{\sqrt d}&#10;\begin{bmatrix}&#10;q_1^\top k_1 &amp; \dots &amp; q_1^\top k_n\\&#10;\vdots &amp; &amp; \vdots \\&#10;q_n^\top k_1 &amp; \dots &amp; q_n^\top k_n&#10;\end{bmatrix}\right) \begin{bmatrix}v_1\\\vdots\\v_n\end{bmatrix},&#10;$$" src="svgs/8d94e59957c56a775aa4d62326b58e70.svg" align="middle" width="314.2365501pt" height="69.72832185pt"/></p>
where the softmax is applied row-by-row, corresponding to each query.

**Multi-head attention block (MAB)**

The multi-head attention block with queries <img alt="$Q$" src="svgs/1afcdb0f704394b16fe85fb40c45ca7a.svg" align="middle" width="12.99542474999999pt" height="22.465723500000017pt"/> and values <img alt="$V$" src="svgs/a9a3a4a202d80326bda413b5562d5cd1.svg" align="middle" width="13.242037049999992pt" height="22.465723500000017pt"/>, and treats the values <img alt="$V$" src="svgs/a9a3a4a202d80326bda413b5562d5cd1.svg" align="middle" width="13.242037049999992pt" height="22.465723500000017pt"/> as keys <img alt="$K$" src="svgs/d6328eaebbcd5c358f426dbea4bdbf70.svg" align="middle" width="15.13700594999999pt" height="22.465723500000017pt"/>. It uses <img alt="$h$" src="svgs/2ad9d098b937e46f9f58968551adac57.svg" align="middle" width="9.47111549999999pt" height="22.831056599999986pt"/> heads to project the queries, keys, and values into <img alt="$h$" src="svgs/2ad9d098b937e46f9f58968551adac57.svg" align="middle" width="9.47111549999999pt" height="22.831056599999986pt"/> different embeddings through linear transforms. Then it applies attention to each of the <img alt="$h$" src="svgs/2ad9d098b937e46f9f58968551adac57.svg" align="middle" width="9.47111549999999pt" height="22.831056599999986pt"/> embeddings, concatenates the resulting features at the end, then adds it to the result of another feed-forward neural network at the end.
<p align="center"><img alt="$$&#10;\begin{align*}&#10;\mathrm{MAB}(Q,V) &amp; =[O_1,\dots,O_h] + \mathrm{ReLU}([O_1,\dots,O_h]W^O)\\&#10;O_i &amp; = \text{Attention}(QW_i^Q, VW_i^K,VW_i^V)&#10;\end{align*}&#10;$$" src="svgs/33ce2bfe2536fdab7bb132ae8246aa14.svg" align="middle" width="384.4197258pt" height="47.48807745pt"/></p>


Note that above <img alt="$\{W_i^Q,W_i^K,W_i^V\}_{i=1}^h$" src="svgs/f9b3db74710c335cd514a1235fb9a5e8.svg" align="middle" width="141.03335894999998pt" height="31.525041899999984pt"/> and <img alt="$W^O$" src="svgs/e2187303a2bc2de82c270978ec834929.svg" align="middle" width="28.16078594999999pt" height="27.6567522pt"/> are trainable parameters. 

**Set attention block (SAB)**

This is just a MAB with self-attention over element of the set.
<p align="center"><img alt="$$&#10;\mathrm{SAB}(X) = \mathrm{MAB}(X,X)&#10;$$" src="svgs/80fcb5ba5c25c8d423d4fe8c8d287818.svg" align="middle" width="170.75344935pt" height="16.438356pt"/></p>


**Induced set attention block (ISAB)**

To reduce the computational complexity of self-attention we replace the values with a set of <img alt="$m$" src="svgs/0e51a2dede42189d77627c4d742822c3.svg" align="middle" width="14.433101099999991pt" height="14.15524440000002pt"/> induced values. These values are determined by attending over <img alt="$X$" src="svgs/cbfb1b2a33b28eab8a3e59464768e810.svg" align="middle" width="14.908688849999992pt" height="22.465723500000017pt"/> with a set of <img alt="$m$" src="svgs/0e51a2dede42189d77627c4d742822c3.svg" align="middle" width="14.433101099999991pt" height="14.15524440000002pt"/> inducing points <img alt="$I$" src="svgs/21fd4e8eecd6bdf1a4d3d6bd1fb8d733.svg" align="middle" width="8.515988249999989pt" height="22.465723500000017pt"/> as query vectors. 
<p align="center"><img alt="$$&#10;\begin{align*}&#10;&#9;\mathrm{ISAB}(X) &amp;= \mathrm{MAB}(X,H),\\&#10;&#9;H &amp; = \mathrm{MAB}(I, X)&#10;\end{align*}&#10;$$" src="svgs/e5a9bbf4fd5dbd156d6448d2e45266de.svg" align="middle" width="181.3470285pt" height="41.09589pt"/></p>


**Pooling by multi-head attention (PMA)**

After applying layers of set attention blocks, we want to pool the output into a set of <img alt="$k$" src="svgs/63bb9849783d01d91403bc9a5fea12a2.svg" align="middle" width="9.075367949999992pt" height="22.831056599999986pt"/> elements. To do so we learn a set of <img alt="$k$" src="svgs/63bb9849783d01d91403bc9a5fea12a2.svg" align="middle" width="9.075367949999992pt" height="22.831056599999986pt"/> seed vectors <img alt="$S$" src="svgs/e257acd1ccbe7fcb654708f1a866bfe9.svg" align="middle" width="11.027402099999989pt" height="22.465723500000017pt"/> and which are queried over the elements <img alt="$X$" src="svgs/cbfb1b2a33b28eab8a3e59464768e810.svg" align="middle" width="14.908688849999992pt" height="22.465723500000017pt"/>.
<p align="center"><img alt="$$&#10;\mathrm{PMA}_k(X) = \mathrm{MAB}(S,X)&#10;$$" src="svgs/314c2b35f647c0a9e3dcedfb4f1eb65b.svg" align="middle" width="180.43955325pt" height="16.438356pt"/></p>

#### Example

In the below example we try to regress onto the element in each dataset with the largest norm.

![max_value](examples/max_value.png)

#### References

[1] Lee, J., Lee, Y., Kim, J., Kosiorek, A., Choi, S., and Teh, Y.W. (2019). Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks. In International Conference on Machine Learning, pp. 3744–3753.

[2] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, Ł., and Polosukhin, I. (2017). Attention is All you Need. In Advances in Neural Information Processing Systems 30, I. Guyon, U.V. Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, eds. (Curran Associates, Inc.), pp. 5998–6008.
