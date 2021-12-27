





# Gaph Neural News Recommendation with Long-term and Short-term Interest Modeling



Linmei Hu a, Chen Li a, Chuan Shi  a,∗, Cheng Yang a, Chao Shao b 

a: Beijing University of Posts and Telecommunications, Beijing, China

b：Alibaba Group, Hangzhou, China

Preprint submitted to Journal of Information Processing and ManagementNovember 11, 2019(CCF-B)



## 前言

贡献总结如下:

1)本文提出了一种基于长期和短期用户兴趣建模的图神经新闻推荐模型GNewsRec。

2)通过构建异构的用户-新闻-主题图对用户-项目交互进行建模，减少了用户-项目交互的稀疏性。然后利用图卷积网络在图上传播高阶信息，学习用户嵌入和新闻嵌入。

3)在真实数据集上的实验结果表明，该模型在新闻推荐方面的性能明显优于最新的方法。

## GNewsRec

GNewsRec主要包含三个部分:用于文本信息提取的CNN，用于长期用户兴趣建模和新闻建模的GNN，用于短期用户兴趣建模的基于注意力的LSTM模型。

![截屏2021-04-29 下午4.03.11](https://i.loli.net/2021/04/29/edW8soYR9tbQHwa.png)

CNN：从新闻文本和新闻简介中提取文本特征

GNN：通过所有用户的历史点击记录构建user-news-topic图，应用GNN编码得到高阶结构信息。合并的潜在主题信息可以缓解用户-项目的稀疏性，因为用户点击次数少的新闻项目可以通过主题桥聚合更多的信息。由此学习到的用户表征作为用户长期兴趣，此外还得到候选新闻的特征表示。

Attention Based LSTM：对当前用户近期阅读历史的短期兴趣进行建模



文章将新闻的标题和概要(新闻页面内容中给定的实体E及其实体类型C)视为特征。



### Text Information Extractor

两个CNN分别对新闻title和profile进行文本信息的提取，二者拼接为新闻的文本特征表示，得到一个d维向量



### Long-term User Interest Modeling and News Modeling

#### Heterogeneous User-News-Topic Graph

![截屏2021-04-29 下午4.19.50](https://i.loli.net/2021/04/29/LU8ZyjbOgwCxGNp.png)

构建异构图，G = （V，R），V和R分别为节点和边的集合。节点类型包括用户U、新闻item I、主题 Z。

user-item节点：如果用户u点击了该条新闻d，则构建用户到新闻的边，则为 $ y_{u,d} = 1$, 对于每个新闻文档d，我们可以通过LDA得到其主题分布,新闻文档d与概率最大的主题z进行连接。

为了进行测试，我们可以根据估计的LDA模型[30]推断新文档的主题。这样就可以将图中不存在的新文档与所构造的图连接起来，并通过图卷积更新它们的嵌入层。因此，主题信息可以缓解冷启动问题以及用户-项目交互的稀疏性问题

#### GNN for Heterogeneous User-News-Topic Graph

【潜在偏好也可以类似使用图来进行操作】

考虑到候选对用户u和新闻d，使用 U（d）和 Z（d） 分别表示与新闻文档d直接相连的用户集合和主题集合。

每个新闻假设只有一个主题，即｜ Z（d）｜ = 1

真正应用时U（d）的数量是不固定的，为了保证批处理的计算模式固定和更加高效，对每个新闻d统一抽取固定数量的邻居用户节点S（d）而不是抽取所有邻居节点，其中｜S（d）｜ = Lu，如果 | U（d）|< Lu, S(d)可能包含重复值。如果U（d）为空集，S（d）也为空集。

为了表征新闻d的拓扑邻近结构，我们首先计算其所有采样邻居的线性平均组合:

![截屏2021-04-29 下午4.37.26](https://i.loli.net/2021/04/29/vKqCPt85Wm1aHgB.png)

S（d）为新闻d邻居的用户u集合，u和z都为随机初始化的d维向量，文档d由Text Information Extractor提取的文本特征初始化

然后更新候选新闻的嵌入表示

![截屏2021-04-29 下午4.48.51](https://i.loli.net/2021/04/29/B8wSxQymU7vkuO3.png)

这是一个单层的GNN，候选新闻的最终嵌入仅依赖于它的近邻用户节点。



两层的GNN如右图所示，首先通过聚合相邻的新闻嵌入来得到其1阶近邻用户嵌入ul 和主题嵌入z，然后聚合ul和z得到2阶的新闻标准d

一般来说，一个新闻的H阶表示是它在H跳之前的邻居的初始表示的混合。

通过GNN，我们可以得到最终用户和新闻嵌入的高阶信息编码的ul和d。用户嵌入与完整的用户点击历史应该捕捉相对稳定的长期用户兴趣。



### Short-term User Interest Modeling

首先使用一种注意机制来模拟用户最近点击的新闻对候选新闻d的不同影响，得到用户内容级的兴趣嵌入

![截屏2021-04-29 下午4.58.26](https://i.loli.net/2021/04/29/6LvoThrp2cXCWGw.png)

然后应用LSTM得到用户的序列特征表示。

 由于用户的每个点击历史都受到前面的点击历史影响，应用注意力机制于LSTM的每个隐藏单元和当前隐藏单元前面的所有隐藏单元，已获得更加丰富的序列特征Sj，l个特征S由CNN整合得到用户最新l个点击记录的序列特征表示s

拼接用户内容级嵌入和序列级嵌入得到用户的短期兴趣：

![截屏2021-04-29 下午5.02.06](https://i.loli.net/2021/04/29/XfF6iBrAtbQKkLP.png)

最后的用户表征由长期兴趣和短期兴趣拼接而成：

![截屏2021-04-29 下午5.12.45](https://i.loli.net/2021/04/29/pCK8jvOcZNfrFJA.png)

使用DNN预测用户u点击新闻d的概率

![截屏2021-04-29 下午5.13.40](https://i.loli.net/2021/04/29/BouvtODSw37klLJ.png)



## Experiments



数据集：Adressa



![截屏2021-04-29 下午5.21.20](https://i.loli.net/2021/04/29/9PQSHXmstNuEMGd.png)

(1)模型构建了异构的用户-新闻-主题图，通过GNN编码的高阶信息学习到更好的用户和新闻嵌入。

(2)模型不仅考虑了用户的长期利益，还考虑了用户的短期利益。

(3)异构图中包含的主题信息可以更好地反映用户的兴趣，缓解交互中用户-物品的稀疏性问题。用户点击很少的新闻条目仍然可以通过主题聚合相邻信息。





![截屏2021-04-29 下午5.25.27](https://i.loli.net/2021/04/29/YOCMBetUgdpRFzc.png)

消融实验证明以上三点

## ![截屏2021-04-29 下午5.30.19](https://i.loli.net/2021/04/29/GIwNuiQ74R9hqH6.png)

![截屏2021-04-29 下午5.31.02](https://i.loli.net/2021/04/29/vWAERyj4HwaMZ82.png)





## 总结

使用图神经网络的同时还使用主题模型抽象出一类特征



# Graph Neural News Recommendation with Unsupervised Preference Disentanglement

Linmei Hu1, Siyong Xu1, Chen Li1, Cheng Yang1, Chuan Shi1∗ Nan Duan2, Xing Xie2, Ming Zhou2

1School of Computer Science,Beijing University of Posts and Telecommunications, Beijing, China

2Microsoft Research Asia, Beijing, China

ACL2020

## 前言

![截屏2021-05-07 上午11.01.44](/Users/mac/Library/Application Support/typora-user-images/截屏2021-05-07 上午11.01.44.png)

u1-d1-u2的高阶关系表示u1和u2之间的行为相似性，因此u1点击d3后我们可以将d3推荐给u2，而d1-u2-d4则表示d1和d4可能有相似的目标用户。
此外，由于用户的喜好差异很大，他们可能会点击不同的新闻。现实世界的用户新闻互动是由高度复杂的潜在偏好因素产生的。例如，如图1所示，u2可能会在她对娱乐新闻的偏好下点击d1，而选择d4是因为她对政治感兴趣。**在沿图聚合邻域信息时，应考虑邻居节点在不同潜在偏好下的不同重要性**。学习表征能够揭示和解开这些潜在的偏好因素，从而增强新闻推荐的表达性和可解释性

主要贡献：

（1）在本研究中，我们将用户-新闻交互建模为二部图，并提出一种新的无监督偏好解缠的图神经网络新闻推荐模型。我们的模型通过充分考虑用户-新闻交互的高阶连接和潜在的偏好因素来提高推荐性能。

（2）在我们的GNUD模型中，我们设计了一个偏好正则化器来强制每个被解缠的嵌入空间独立地反映一个孤立的偏好，进一步提高了用户和新闻的解缠表示的质量。

（3）在真实数据集上的实验结果表明，该模型的新闻推荐性能明显优于最新的新闻推荐方法。

Disentangled representation learning：**解构表征学习**

## METHOD

### News Content Information Extractor

两个CNN分别对新闻title和profile(新闻标题的实体及相应的实体类别)进行文本信息的提取，二者拼接为新闻的文本特征表示，得到一个d维向量$h_d$



### GNUD框架：偏好分解的图卷积和偏好正则化

![截屏2021-05-07 上午11.34.15](/Users/mac/Library/Application Support/typora-user-images/截屏2021-05-07 上午11.34.15.png)



构建二部图 G = {U , D, E }, U和D分别是用户集合和新闻集合，E是边的集合，每条边e = （u，d）属于E，表示用户u点击了新闻d。

我们的GNUD模型使信息在用户和新闻之间沿着图进行传播，从而捕获用户和新闻之间的高阶关系。此外，GNUD学习了揭示用户新闻交互背后的潜在偏好因素的离散嵌入，增强了表达性和可解释性。

### Graph Convolution Layer with Preference Disentanglement

由二分图得到用户的随机初始化嵌入$h_u$,由文本信息提取器得到新闻嵌入$h_d$,使用图卷积层通过聚合节点u的邻居的特征来学习节点u的表征$y_u$

![截屏2021-05-07 上午11.41.42](https://i.loli.net/2021/05/07/OET2GC15NB7hcRu.png)

考虑到用户的点击行为可能是由不同的潜在偏好因素引起的，我们提出了一个层Conv(·)，使得输出的$y_u$和$y_d$是分解形式的表示。每个分离的向量反映了一个与用户或新闻相关的偏好因素。学习到的解纠缠用户和新闻嵌入可以增强表达性和可解释性。

假设有k类潜在因素，让$y_u = [z_{u,1},z_{u,2},...z_{u_K}]$ , $y_d = [z_{d,1},z_{d2},...z_{d,K}]$，$z_{d,k}\in R^{l_{out}/K},(1<=k<=K),l_{out}是y_u的维度大小$

对于每个用户/物品都构建了K个偏好因素。

#### 用户u的学习过程：以用户为例，为用户建模k个偏好

给定用户节点u，和相邻的新闻节点d : (u, d) ∈ E ，使用子空间特有的投影矩阵Wk将特征向量$h_i∈R^{l_{in}}$映射到第k个偏好相关的子空间:

![截屏2021-05-07 上午11.56.27](https://i.loli.net/2021/05/07/zDqyANhFjkvnIU5.png)

$s_{u,k}$不等于u: $z_{u,k}$ 的第k个分量的最终表示，因为它还没有从邻近的新闻中挖掘任何信息。为了构造$z_{u,k}$，我们需要同时挖掘$s_{u,k}$和邻域特征$s_{u,k}:(u,d)\in E$的信息。



#### Neighborhood routing algorithm.

为了多个角度建模用户/新闻，在建模的时候应该只使用受当前偏好因素影响的新闻，而非考虑所有相关新闻。因此，文中提出了一个近邻路由算法来找到受不同偏好影响的近邻新闻。

邻域路由算法通过迭代分析用户与点击新闻形成的潜在子空间，推断出用户与新闻交互背后的潜在偏好因素。

![截屏2021-05-07 下午12.01.47](https://i.loli.net/2021/05/07/YNSVxr43P9fj8oZ.png)

开始初始化$z_{u,k}$ = $s_{u,k}$，$r_{d,k}$表示在偏好k的子空间下用户u和点击新闻d的相似度【由于偏好k导致用户u点击新闻d的概率】

每次迭代计算$r_{d,k}$,把用户和新闻分别在k个子空间进行比较

![截屏2021-05-07 下午6.00.24](https://i.loli.net/2021/05/07/LShPltcrxqINXBo.png)

更新目标向量

![截屏2021-05-07 下午9.00.05](https://i.loli.net/2021/05/07/MEqrHdtZ7c9Iw2K.png)

最后得到的$z^T_{u,k}$是用户u在子空间k下的最后表示，得到 $y_u = [z_{u,1},z_{u,2}...,z_{u.K}]$

为了从高阶邻域中获取信息并学习高阶特征，叠加多个图卷积层，得到最后用户表示 $y_u^{(L)} \in R^{K \triangle\  n }$和新闻表示$y_d^{(L)}$。

卷积层只是一个说法，指的是整个运算过程，而且这个偏好是无监督信号，是模型自己要去学习的。

### Preference Regularizer

我们希望每个解缠子空间都能独立地反映一个孤立的潜在偏好因子。由于训练数据中没有明确的用户偏好标记，本文还设计了一种新的偏好正则化器，使用最大化衡量两个变量的依赖性的**互信息**来对分解表征进行约束。

最大化**互信息**具体表现为：给定用户u在第k(1≤k≤K)潜在子空间中的表示，偏好调节器$P(k|z_{u,k})$估计了$z_{u,k}$属于第k个子空间的概率

![截屏2021-05-07 下午12.07.14](https://i.loli.net/2021/05/07/NE47So3kMUOxbZY.png)

【此处不是模型的计算公式，而是正则项的由来】

### Model Training

增加全连接层得到用户的表示$y'_u = W^{(L+1)T}y_u^{(L)}+ b^{L+1}$,通过点乘计算分数 $\hat s <u,d> =\acute{y}_u^T\acute{y}_d $，根据真实标签$\hat y_{u,d}$使用的目标函数为：

![截屏2021-05-07 下午5.29.35](https://i.loli.net/2021/05/07/Zed3tMUubPTn2Rx.png)

为u和d加上偏好正则化项：

![截屏2021-05-07 下午5.32.22](https://i.loli.net/2021/05/07/2jF6dDvsnLNBhZa.png)

【此处应该为分别最大化用户和新闻的k个子空间之间的互信息】

完整的训练损失为：



![截屏2021-05-07 下午5.32.37](https://i.loli.net/2021/05/07/HKyk9oAPbNiQwhV.png)

对于每个正样本(u, d)，我们从u未观察到的阅读历史中抽取一个负样本进行训练，λ是平衡系数。η为正则化系数，∥Θ∥为所有可训练参数。

值得注意的是，对于新物品而言，可以视为图中的孤立点，其表示可以通过单纯的内容特征表示，也可以通过前面提到的分解方式跟不同的隐含偏好做计算。



#### 互信息

互信息衡量两个变量的依赖程度，可以理解为给定其中一个变量Y，可以多大程度的减少另一个变量X的不确定性，具体为![截屏2021-05-10 下午8.00.19](https://i.loli.net/2021/05/10/GhOlLIMHnVeXFvJ.png)

最大化互信息条件，就是最大化两个随机事件的相关性。







## Experiments



数据集：Adressa



![截屏2021-05-07 上午11.49.25](https://i.loli.net/2021/05/07/ILVbGTjmK7hxANr.png)

### Case Study

![截屏2021-05-07 下午10.34.28](https://i.loli.net/2021/05/07/VmzFranUlQkDCy2.png)

将用户u的表示分解为K = 7个子空间，并随机抽取2个子空间。

对于每一个，我们可视化用户u最关注的头条新闻(概率是rd,k大于阈值)。

如图3所示，不同的子空间反映了不同的偏好因素。

例如，一个子空间(蓝色显示)是与“能源”相关的，因为前两个新闻包含了关键词，如“石油工业”，“氢气”和“风能”。

另一个子空间(绿色部分)可能表示对"健康饮食"的潜在偏好因素，因为相关新闻中包含"健康"、"维生素 "和"蔬菜"等关键词。

关于家的新闻d3在两个子空间中概率都很低。它不属于这两个首选项中的任何一个



### Parameter Analysis

![截屏2021-05-07 下午10.38.41](https://i.loli.net/2021/05/07/pPAkzxuOmbqQr1v.png)

![截屏2021-05-07 下午10.38.57](https://i.loli.net/2021/05/07/pKauSROcbHIFNjA.png)

## 总结

论文核心是图神经网络+偏好分解，和我设想的多向量表示用户的思想一致。

# Graph Enhanced Representation Learning for News Recommendation

Suyu Ge 、Chuhan Wu 、Fangzhao Wu、 Tao Qi、 Yongfeng Huang

In Proceedings of The Web Conference 2020 (WWW ’20), April 20–24, 2020, Taipei, Taiwan. ACM, New York, NY, USA, 7 pages. https://doi.org/10.1145/3366423. 3380050

## 前言

![截屏2021-05-08 下午5.53.39](https://i.loli.net/2021/05/08/Y2orhU1GyecJs9P.png)

同一个用户查看的新闻为邻居新闻，共同点击某个新闻的多个用户为邻居用户。比如新闻n1和n5是相邻的，u1和u2是相邻用户。由此提出：

1. 新闻可以通过邻居新闻增强表示
2. 用户可以通过邻居用户增强表示

## GERL



![截屏2021-05-08 下午6.04.18](https://i.loli.net/2021/05/08/3Cgj2kYO6xNceKo.png)

### Transformer for Context Understanding

![截屏2021-05-08 下午6.05.48](https://i.loli.net/2021/05/08/4cyZK6wNBlJaAHb.png)

通过transformer来得到新闻标题和主题的上下文表征，但作者实验使用原始的transformer的效果是次优的，因此进行简化。

1.将标题序列[w1,w2,...,wM]嵌入到低维空间[e1, e2, ..., eM ].

2.基于单词级别的多头自注意力机制，通过词汇间的相互作用学习新闻的表征

![截屏2021-05-08 下午6.10.57](https://i.loli.net/2021/05/08/RcfwjBxaYyQ8Go6.png)

第i个单词的多头表示为$h_i = [h^1_i;h^2_i;...h_i^N]$

3.加入dropout防止过拟合

4.通过一个attention网络对不同单词的重要性进行建模：

![截屏2021-05-08 下午6.14.35](https://i.loli.net/2021/05/08/1eZ3cMzRTCaxjPE.png)

最后新闻表示为 $v_t = \sum^M_{i=1}\beta^w_i h_i$

5.新闻主题通过词嵌入进行建模$v_p$

6.最终的新闻表示为新闻标题和新闻主题的嵌入的拼接 $v = [v_t;v_p]$

### One-hop Interaction Learning

主要用来学习

(1) Candidate news semantic representations; 

(2) Target user semantic representations; 

(3) Target user ID representations.

#### Candidate News Semantic Representations

候选新闻采用transformer建模表示为$n^O_t$

#### Target User Semantic Representations

通过注意力机制根据用户的点击历史建模目标用户的偏好，假设用户u的历史点击新闻列表为$【n_1,n_2...n_K】$，经过transformer后编码为$【v_1,v_2...v_K】$，注意力网络权重表示为:


![截屏2021-05-08 下午7.54.17](https://i.loli.net/2021/05/08/q2dBOwfab9D8xzW.png)

最后one-hop的用户语义表示为 $u^O_t = = \sum^K_{i=1}\beta^n_iv_i$

#### Target User ID Representations

用户IDs代表独一无二的用户，我们为每一个用户学习一个向量$u^O_e$



### Two-hop Graph Learning

两跳图学习模块从交互图中挖掘邻居用户与新闻之间的相关性。此外，对于一个给定的目标用户，邻居用户通常与他/她有不同程度的相似度。邻居之间的新闻也存在同样的情况。为了利用这种相似性，我们用一个图关注网络[22]来聚合邻居新闻和用户信息。

【此处类似于NPA的新闻级别的attention和单词级别的attention】

#### Neighbor User ID Representations

已知目标用户的邻居用户$[u_{n1},u_{n2}...u_{nD}]$,ID嵌入为$[m_{n1},m_{n2}...m_{nD}]$,通过注意力机制聚合邻居用户的ID嵌入补充目标用户ID表示：

![截屏2021-05-08 下午8.07.10](https://i.loli.net/2021/05/08/k6pyS35GlgA7zKI.png)

two-hop的邻居用户ID表示为：$u^T_e = \sum^D_{i=1}\beta^u_im_{ui}$

#### Neighbor News ID Representations

同上得到two-hop的邻居新闻ID表示$n^T_e$

#### Neighbor News Semantic Representations

同上利用邻居新闻的文本内容经过transformer,attention得到two-hop的邻居新闻语义表示$n^T_t$


### Recommendation and Model Training

目标用户表示为： $u = [u^O_t:u^O_e:u^T_e]$

候选新闻表示为： $n = [n^O_t:n^O_e:n^T_e]$

分数计算为：$\hat y = u^Tn$

目标函数：![截屏2021-05-08 下午8.17.23](https://i.loli.net/2021/05/08/T3rlJnkvFRLHsSW.png)



## EXPERIMENTS

数据集为微软新闻

![截屏2021-05-17 上午11.26.17](https://i.loli.net/2021/05/17/MN2pYBKd4vRIwgr.png)



![截屏2021-05-08 下午8.23.52](https://i.loli.net/2021/05/08/fdnKIR6FWeNbcYv.png)

### Effectiveness of Graph Learning

![截屏2021-05-08 下午8.33.25](https://i.loli.net/2021/05/08/yN5w8YJqu1lTpCA.png)



### Ablation Study on Attention Mechanism

![截屏2021-05-08 下午8.31.29](https://i.loli.net/2021/05/08/Np73Wg2Xo5ORYFU.png)

### Hyperparameter Analysis

![截屏2021-05-08 下午8.30.31](https://i.loli.net/2021/05/08/XSN6idjfL7WOYQK.png)





## 总结



# MVL: Multi-View Learning for News Recommendation





Tokala Yaswanth Sri Sai Santosh,Avirup Saha，Niloy Ganguly

Indian Institute of Technology Kharagpur【印度理工学院】

SIGIR ’20, July 25–30, 2020, Virtual Event, China



本文提出了一种基于内容视图和用户-新闻交互图视图的多视图学习(MVL)新闻推荐框架。

在内容视图中，我们使用新闻编码器从不同的信息(如标题、正文和类别)学习新闻表示。我们从他/她浏览的新闻中获得用户代表，条件是要推荐的候选新闻文章。

在图视图中，通过对不同用户和新闻之间的交互建模，我们提出了一种图神经网络来捕获用户-新闻、用户-用户和新闻-新闻的相关性。此外，我们提出将注意机制纳入到图神经网络中，以模拟这些交互的重要性，以提高用户和新闻的信息表示学习。



# MVL



### Content View [𝑛𝑐 , 𝑢𝑐 ]

该模块的核心是一个新闻编码器，用于学习新闻文章的表示，并根据用户浏览的历史新闻，以推荐的新闻文章为条件，对用户进行表示。

#### NewsEncoder:通过注意力机制合并新闻不同视角的特征：标题、正文、类别

Title encoder：先通过嵌入层得到词嵌入，然后经过CNN捕获局部信息，最后通过注意力机制为建模词的重要性，得到新闻标题的表示$r^t$

Body and Category Encoder:同上得到正文的表示$r^b$和类别的表示$r^c$

Attentive Network：分别计算三类特征的注意力权重

标题权重计算

![截屏2021-05-20 下午9.35.31](https://i.loli.net/2021/05/20/4lXVeKqCFWtkLG8.png)

同理得到正文和类别的权重，聚合得到新闻的表征：

![截屏2021-05-20 下午9.37.12](https://i.loli.net/2021/05/20/VMcYXDu7a5BA8TU.png)

#### Obtaining user representation 

用户点击历史为$[n_1,n_2,...,n_L]$,候选新闻为$n_x$

以候选新闻为条件模拟用户最近点击新闻的不同影响的用户级关注网络计算权重：

![截屏2021-05-20 下午9.41.42](https://i.loli.net/2021/05/20/ndm8Nq74l39SOU1.png)



用户点击的新闻表征汇聚得到用户表示：

![截屏2021-05-20 下午9.40.02](https://i.loli.net/2021/05/20/yph86I3YPNFOwDf.png)



Graph View [𝑢𝑔 , 𝑛𝑔 ] 
在内容视图中，我们试图隐式捕获用户使用其浏览文章的第一阶交互。

为了直接获取用户的二阶交互以及新闻的一阶和二阶交互，我们提出了一种层次注意力图神经网络来学习表示。首先通过内容视图的嵌入来表示每个节点(用户或新闻)，增强静态用户-新闻双部图。然后利用该图对图中用户与新闻的一级和二级交互(跳)建模，挖掘用户-用户和新闻-新闻的相关性

#### Learning 𝑢𝑔

让用户𝑢浏览的新闻是[𝑛1，𝑛2，…,𝑛𝑃)和浏览新闻𝑛𝑖的用户为[𝑢𝑖1,𝑢𝑖2,…,𝑢𝑖𝐾]

建模不同用户对同一新闻的重要性：

![截屏2021-05-20 下午9.47.52](https://i.loli.net/2021/05/20/qpchXQMzV9lPEND.png)

![截屏2021-05-20 下午9.54.58](https://i.loli.net/2021/05/20/NpUeKaBkDHtV2ym.png)

得到新闻节点的嵌入后，建模用户u邻居新闻节点的重要性：


![截屏2021-05-20 下午9.57.42](https://i.loli.net/2021/05/20/btuH6VT91oxANGy.png)

得到基于的图的用户表征$u_g$

![截屏2021-05-20 下午9.57.51](https://i.loli.net/2021/05/20/GMkDF7YQhNwqZf6.png)



同理学习基于图的新闻表征n_g



最终的用户表示为：【uc,ug】

最终的新闻表示为：【n c,ng】

点击概率为：

![截屏2021-05-20 下午10.17.38](https://i.loli.net/2021/05/20/PnsFCql4oNcBMHi.png)

# 实验

数据集为Adressa1-week和Adressa-10week，使用的词嵌入为word2vec

![截屏2021-05-20 下午10.01.55](https://i.loli.net/2021/05/20/w8kEVOxBAgeqbPu.png)



消融实验

![截屏2021-05-20 下午10.04.53](https://i.loli.net/2021/05/20/FrdEOsbMAQtcgzN.png)

内容视图中合并各种信息的有效性



![截屏2021-05-20 下午10.05.30](/Users/mac/Library/Application Support/typora-user-images/截屏2021-05-20 下午10.05.30.png)

各种注意力的有效性





![截屏2021-05-20 下午10.06.22](/Users/mac/Library/Application Support/typora-user-images/截屏2021-05-20 下午10.06.22.png)

比较了MVL与仅涉及零阶或同时涉及零阶和一阶相互作用的变体的性能。我们观察到，随着深度的增加，MVL的性能可以持续提高。这是因为图的一阶信息包含用户和商品之间的交互，二阶信息可以显示用户-用户和商品-商品之间的相关性。因此，随着深度的增加，可以吸收更多的信息，有利于用户和项目表示学习。



![截屏2021-05-20 下午10.07.19](https://i.loli.net/2021/05/20/tTCoNU6OWJvnie1.png)

图中注意力的重要性





# 总结

内容实质上就是NPA的简化版，只是没有使用所谓的个性化查询向量

图卷积的过程就是分层GAT，计算用户时先针对用户点击的每个新闻进行一次GAT，然后根据用户的点击历史即邻居新闻做GAT，同理计算新闻先以点击该新闻的每个用户对其点击历史做GAT，再根据新闻的邻居用户做GAT

