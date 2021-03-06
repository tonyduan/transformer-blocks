### Transformer Models

Last update: April 2020.

---

Implementations of blocks for attention-based models; the encoder/decoder blocks of the original Transformer [1] and permutation-equivarant blocks of the Set Transformer [2]. The Annotated Transformer [3] proved a useful reference.

**Scaled Dot-Product Attention**

For set of queries <img alt="$Q$" src="svgs/1afcdb0f704394b16fe85fb40c45ca7a.svg" align="middle" width="12.99542474999999pt" height="22.465723500000017pt"/>, keys <img alt="$K$" src="svgs/d6328eaebbcd5c358f426dbea4bdbf70.svg" align="middle" width="15.13700594999999pt" height="22.465723500000017pt"/>, and values <img alt="$V$" src="svgs/a9a3a4a202d80326bda413b5562d5cd1.svg" align="middle" width="13.242037049999992pt" height="22.465723500000017pt"/>, we define:
<p align="center"><img alt="$$&#10;\text{Attention}(Q,K,V) = \text{Softmax}\left(\frac{1}{\sqrt{d_k}}QK^\top\right)V.&#10;$$" src="svgs/f8f408989954b0201f9cb8f8143142a7.svg" align="middle" width="336.35182844999997pt" height="39.452455349999994pt"/></p>


The interpretation is that each query <img alt="$q_k$" src="svgs/acc433299a99053a1abda8186a07965d.svg" align="middle" width="14.604341399999988pt" height="14.15524440000002pt"/> corresponds to a different feature, which is constructed as a convex combination of values <img alt="$v_k$" src="svgs/eaf0887cdc4cb5f8e69a7796f143c3eb.svg" align="middle" width="15.23409524999999pt" height="14.15524440000002pt"/>, where the weights are given by the dot products between the query and the keys <img alt="$k_k$" src="svgs/1271cba53526e0c04aa67fe6b06300be.svg" align="middle" width="15.82390589999999pt" height="22.831056599999986pt"/>. To see this, we can expand (note each matrix consists of *row vectors*, not column vectors):
<p align="center"><img alt="$$&#10;\text{Softmax}\left(\frac{1}{\sqrt d}&#10;\begin{bmatrix}&#10;q_1^\top k_1 &amp; \dots &amp; q_1^\top k_n\\&#10;\vdots &amp; &amp; \vdots \\&#10;q_n^\top k_1 &amp; \dots &amp; q_n^\top k_n&#10;\end{bmatrix}\right) \begin{bmatrix}-\ v_1\ -\\\vdots\\-\ v_n\ -\end{bmatrix} = &#10;\begin{bmatrix}-\ \sum_{i=1}^n w_i^{(1)}v_i -\\ \vdots \\ -\ \sum_{i=1}^n w_i^{(n)}v_n\ -\end{bmatrix},&#10;$$" src="svgs/2ccfcb2061cb8c5781cfe11355e8a20c.svg" align="middle" width="527.0764224pt" height="78.9048876pt"/></p>


where the softmax is applied row-by-row, corresponding to a softmax per query. [Note: In the above <img alt="$w_i^{(j)}$" src="svgs/457aecf08c6a732074bee5cb6c63f1d8.svg" align="middle" width="28.58936519999999pt" height="34.337843099999986pt"/> corresponds to the weights over <img alt="$v_1,\dots,v_n$" src="svgs/ee7eaced4e0a124d979846e13ba6d445.svg" align="middle" width="67.9660311pt" height="14.15524440000002pt"/> that arise from softmax due to the <img alt="$j$" src="svgs/36b5afebdba34564d884d347484ac0c7.svg" align="middle" width="7.710416999999989pt" height="21.68300969999999pt"/>-th query vector <img alt="$q_j$" src="svgs/afb133b4c3a4f1dc590243e9cef58b5b.svg" align="middle" width="13.44282059999999pt" height="14.15524440000002pt"/>.]



We normalize by <img alt="$\frac{1}{\sqrt d_k}$" src="svgs/7e4778c404d57e72596a666d4b00aba4.svg" align="middle" width="24.828348599999995pt" height="27.77565449999998pt"/> (the dimensionality of the queries and keys) to prevent the dot products <img alt="$q_i^\top k_j$" src="svgs/3817f7dd1bbbf640839560cd731979e0.svg" align="middle" width="33.68638844999999pt" height="27.91243950000002pt"/> from blowing up in value and thereby vanishing gradients of the softmax. [Note: Recall that if <img alt="$q \sim N(0, I)$" src="svgs/bec5a0818a78fe9cb64487ed59215094.svg" align="middle" width="81.67217849999999pt" height="24.65753399999998pt"/> and <img alt="$k\sim N(0,I)$" src="svgs/02eeb1563a72b89acb72200cd566dd79.svg" align="middle" width="82.8194532pt" height="24.65753399999998pt"/> then <img alt="$q^\top k\sim N(0, d_k)$" src="svgs/4eb4f5aa82a38184e9c8cc444e34db12.svg" align="middle" width="109.97139779999999pt" height="27.91243950000002pt"/>.]

**Multi-Head Attention**

Multi-head attention projects <img alt="$Q,V,K$" src="svgs/a802f68e319c5b6b9ff81cfceadb03ae.svg" align="middle" width="53.24649329999999pt" height="22.465723500000017pt"/> using <img alt="$h$" src="svgs/2ad9d098b937e46f9f58968551adac57.svg" align="middle" width="9.47111549999999pt" height="22.831056599999986pt"/> heads through linear transforms. Then it applies attention to each of the <img alt="$h$" src="svgs/2ad9d098b937e46f9f58968551adac57.svg" align="middle" width="9.47111549999999pt" height="22.831056599999986pt"/> embeddings, concatenates the resulting features at the end, then sends the result through another linear transform.
<p align="center"><img alt="$$&#10;\begin{align*}&#10;\mathrm{MultiHeadAttn}(Q,V) &amp; = \mathrm{Concat}(O_1,\dots,O_h)W^O\\&#10;\mathrm{where}\ O_i &amp; = \text{Attention}(QW_i^Q, VW_i^K,VW_i^V)&#10;\end{align*}&#10;$$" src="svgs/a51296ad6a6593fe5def10dab4d9001d.svg" align="middle" width="403.2638775pt" height="47.48807745pt"/></p>



Note that above <img alt="$\{W_i^Q,W_i^K,W_i^V\}_{i=1}^h$" src="svgs/f9b3db74710c335cd514a1235fb9a5e8.svg" align="middle" width="141.03335894999998pt" height="31.525041899999984pt"/> and <img alt="$W^O$" src="svgs/e2187303a2bc2de82c270978ec834929.svg" align="middle" width="28.16078594999999pt" height="27.6567522pt"/> are trainable parameters. 

**Language Model**

In this repository we implement a simple language model using the WikiText-2 dataset. For this task given a corpus we try to learn the distribution
<p align="center"><img alt="$$&#10;p(x^{(i)}\ |\ x^{(i-L)},\dots,x^{(i-1)}) \sim \mathrm{Categorical}&#10;$$" src="svgs/093d3388d0bc9e943d82a4048faf2752.svg" align="middle" width="287.89691369999997pt" height="19.526994300000002pt"/></p>
using a set of Transformer encoder blocks followed by an attention-based pooling layer. Note that the English vocabulary has roughly 30k tokens in size.

#### References

[1] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, Ł., and Polosukhin, I. (2017). Attention is All you Need. In Advances in Neural Information Processing Systems 30, I. Guyon, U.V. Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, eds. (Curran Associates, Inc.), pp. 5998–6008.

[2] Lee, J., Lee, Y., Kim, J., Kosiorek, A., Choi, S., and Teh, Y.W. (2019). Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks. In International Conference on Machine Learning, pp. 3744–3753.

[3] G. Klein, Y. Kim, Y. Deng, J. Senellart, & A. Rush, OpenNMT: Open-Source Toolkit for Neural Machine Translation. Proceedings of ACL 2017, System Demonstrations (Vancouver, Canada: Association for Computational Linguistics, 2017), pp. 67–72.