# Neural News Recommendation with Multi-Head Self-Attention

Chuhan Wu,Department of Electronic Engineering, Tsinghua University, Beijing 100084, China

 Fangzhao Wu, Microsoft Research Asia, Beijing 100080, China

Suyu Ge, Department of Electronic Engineering, Tsinghua University, Beijing 100084, China

Tao Qi, Department of Electronic Engineering, Tsinghua University, Beijing 100084, China

Yongfeng Huang,Department of Electronic Engineering, Tsinghua University, Beijing 100084, China

Xing Xie ,Microsoft Research Asia, Beijing 100080, China



**publication**：Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP

Cited times: 14



**propose a neural news recommendation approach with multi-head self-attention (NRMS).**

## NRMS



![截屏2020-10-21 下午9.00.44](https://i.loli.net/2020/10/21/6PdbEXtOrCFSkmh.png)

### News Encoder

New encoder主要是把CNN替换为一个word-level multi-head self-attention network ,原因是CNN只能捕捉上下文之间的关系，而多头注意力可以捕捉一个词序列的远距离的信息交互，如首尾单词的交互。

然后再使用additive word attention network（类似于NPA中使用ID得到一个查询向量），此处可能是随机初始化，为一个新闻标题的不同单词分配权重。



User Encoder也一样，使用a  news-level multi-head self-attention network，捕捉不同新闻的远距离交互

然后再使用additive word attention network（类似于NPA中使用ID得到一个查询向量），此处可能是随机初始化，为用户点击记录的不同新闻分配权重。

1⃣️word embedding ：Denote a news title with M words as $$[w_1,w_2,...,w_M]$$,converted into a vector sequence $$[e_1,e_2,...,e_M]$$

2⃣️a  word-level  multi-head self-attention network 【取代CNN】

 Forexample,  in  the  news  title  “Rockets  Ends  2018with  a  Win”,  the  interaction  between  “Rockets”and “Win” is useful for understanding this news,and  such  long-distance  interactions  usually  can-not be captured by CNN.

In addition, a word may interact  with  multiple  words  in  the  same  news.

The representation of the $i_{th}$ word learned by the $$k_{th}$$ attention head is computed as:


$$
α^k_{i,j} = \frac{exp(e^T_iQ^w_ke_j)} {\sum^M_{m=1}exp(e^T_iQ^w_ke_j)},(1)
$$

$$
h^w_{i,k} = V^w_k(∑^M_{j=1} α^k_{i,j}e_j),(2)
$$

$Q^w_k$ and $V^w_k$ are the projection parameters in the $k_{th}$ self-attention head, and $α^k_{i,j}$ indicates the relative importance of the interaction between the $i_{ith}$ and $j_{th}$ words

The multi-head representation $ h^w_i$ of  the $i_{th}$ word is the concatenation of the representations produced by $h$ separate self-attention heads, i.e. $h^w_i= [ h^w_{i,1}; h^w_{i,2};...h^w_{i,h}]$

3⃣️an additive word attention network

use attention mechanism to select important words in news  titles  for  learning  more  informative  news representations

The  attention  weight $α^w_i$ of  the $i_{th}$ word in a news title is computed as:

$a^w_i=q^T_w tanh(V_w×h^w_i+v_w),(3)$

$α^w_i = \frac{exp(a^w_i)}{\sum^M_{j=1}exp(a^w_j)},(4)$

where $V_w$ and $v_w$ are projection parameters, and $q^w$ is the query vector. 

The final representation of a news is the weighted summation of the contex-tual word representations, formulated as: $ r = \sum^M_{i=1}α^w_ih^w_i.(5)$

### User Encoder

1⃣️a  news-level multi-head self-attention network

a news article may interact with multiple news articles browsed by the same user

apply multi-head self-attention to enhance the representations of news by capturing their interactions.

The representation of the $i_{th}$ news learned by the $k_{th}$ attention head is formulated as follows:
$$
β^k_{i,j} = \frac{exp(r^T_iQ^n_kr_j)}{\sum^M_{m=1}exp(r^T_iQ^n_kr_m)},(6)
$$

$$
h^n_{i,k} = V^n_k(\sum^M_{j=1}β^k_{i,j}r_j),(7)
$$



where $Q^n_k$ and $V^n_k$  are parameters of the $k_{th}$ newsself-attention head, and $β^k_{i,j}$ represents the relative importance of the interaction between the $j_{th}$ and the $k_{th}$ news.  

 The  multi-head  representation  of the $i_{th}$ news is the concatenation of the representations output by h separate self-attention heads,i.e.,

$h^n_i= [h^n_{i,1};h^n{i,2};...;h^n{i,h}]$

2⃣️an additive news attentionnetwork

Different  news  may  have  different  in-formativeness in representing users

apply the additive attentionmechanism to select important news to learn moreinformative  user  representations

The  attention weight of the $i_{th}$ news is computed as: 
$$
a^n_i=q^T_ntanh(V_n×h^n_i+v_n),(8)
$$

$$
α^n_i = \frac{exp(a^n_i)}{\sum^N_{j=1}exp(a^n_j)},(9)
$$

whereVn,$$v_n$$ and $q_n$ are  parameters  in  the  attention  network,  and N is  the  number  of  the browsed news. 

The final user representation is theweighted summation of the representations of thenews browsed by this user, which is formulated as:

$u=\sum^N_{i=1}α^n_ih^n_i.(10)$

### Click Predictor

Denote the representation of a candidate news $D^c$  as  $r^c$.

the  clickprobability score $y$ is computed by the inner product of the user representation vector and the newsrepresentation  vector,  i.e.,$$ˆy=u^Tr_c$$. 

## Experiments

![截屏2020-10-22 上午10.00.06](https://i.loli.net/2020/10/22/yNP8V9FRASOK6tG.png)

