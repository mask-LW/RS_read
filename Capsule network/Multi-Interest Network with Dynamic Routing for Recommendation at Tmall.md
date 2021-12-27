

#  Multi-Interest Network with Dynamic Routing for Recommendation at Tmall

在天猫的推荐过程中，推荐系统也被拆分为召回和排序阶段。

![](https://upload-images.jianshu.io/upload_images/4155986-8a65396dfb13c5ab.png?imageMogr2/auto-orient/strip|imageView2/2/w/1240)

天猫个性平台检索用户近期行为，发送到User Interest Extractor，该提取器是实现MIND将用户行为转化为多种用户兴趣的主要模块。随后，Recall Engine搜索向量最接近用户兴趣的条目，它们可以在15毫秒内完成从数以亿计商品库中选择数千个候选商品。

由不同兴趣触发的项合并在一起作为候选项，并根据它们与用户兴趣的相似性进行排序，在项目范围和系统的响应时间之间进行权衡，排名前1000的候选项目通过Ranking Service进行评分，RankingService预测具有一系列特性的点击率。



本文重点关注召回阶段的算法。召回阶段的目标是得到数千个跟用户兴趣紧密相关的商品候选集。

目前的大多数算法仅仅将用户的兴趣表示成单个的Embedding，这是不足以表征用户多种多样的兴趣的，同时容易造成头部效应。因此本文提出了MIND，同时生成多个表征用户兴趣的Embedding，来提升召回阶段的效果。

## 问题概述

召回阶段的目标是对于每个用户u∈U的请求，从亿级的商品池I中，选择成百上千的符合用户兴趣的商品候选集。每条样本可以表示成三元组（I<sub>u</sub>,P<sub>u</sub>,F<sub>i</sub>)，其中I<sub>u</sub>是用户u历史交互过的商品集合，P<sub>u</sub>是用户画像信息，比如年龄和性别，F<sub>i</sub>是目标商品的特征，如商品ID、商品品类ID。

那么MIND的核心任务是将用户相关的特征转换成一系列的用户兴趣向量：
$$
V_u = f_{user}(I_u,P_u)
$$

$$
V_u = (\vec{v}^1_u,...\vec{v}_u^K) \in R^{d * K}表示一个用户的向量表示，k个d维向量
$$
目标商品：
$$
\vec{e}_i = f_{item}(F_i),\vec{e}_i \in R^{d*1}
$$
当得到用户和商品的向量表示之后，通过如下的score公式计算得到topN的商品候选集：
$$
f_{score}(V_u,\vec{e}_i) = \underset {1\leq k \leq K} {max}\vec{e}^T_i\vec{v}^T_u
$$


## MIND

![](https://i.loli.net/2020/11/23/MTrmhvlJoRUsIS5.png)

## Embedding Layer & Pooling Layer

如上图，MIND的输入中包含三部分，用户的画像信息P<sub>u</sub>、用户历史行为I<sub>u</sub>和目标商品F<sub>i</sub>。每个部分都包含部分的类别特征，类别特征会转换为对应的embedding。

对用户画像信息部分来说，不同的embedding最终拼接在一起。

对于用户历史行为I<sub>u</sub>中的商品和目标商品F<sub>i</sub>来说，商品ID、品牌ID、店铺ID等转换为embedding后会经过avg-pooling layer来得到商品的embedding表示。

**此处只是单纯拼接ID特征，没有文本的序列特征，与新闻推荐不一样**

## Multi-Interest Extractor Layer

将与用户不同兴趣相关的所有信息压缩到一个表示向量中来表示一个用户的兴趣是不足以的，因此用多个向量表示用户兴趣，由向量的角度考虑到使用胶囊网络。

MIND中的Multi-Interest Extractor Layer，与胶囊网络主要有两个地方不同：

1）在胶囊网络中，每一个输入向量和输出向量之间都有一个单独的仿射矩阵，但是MIND中，仿射矩阵只有一个，所有向量之间共享同一个仿射矩阵。主要原因是用户的行为数量长度不同，使用共享的仿射矩阵不仅可以减少参数，同时还能对应另一处的改变，即不同用户输出向量的个数K是基于他历史行为长度自适应计算的：
$$
K_u^， = max(1,min(K,log_2(I_u)))
$$
上面基于用户历史行为长度自适应计算输出向量个数K'的策略，对于那些交互行为较少的用户来说，可以减少这批用户的存储资源。(K为兴趣胶囊的数目，为超参数)

在初始动态路由中，我们使用固定的双线性映射矩阵S而不是单独的双线性映射矩阵。一方面，用户行为是可变长度的，天猫用户从几十个到几百个不等，因此使用固定双线性映射矩阵是可推广的。另一方面，我们希望兴趣胶囊在同一个向量空间中，但不同的双线性映射矩阵将利息胶囊映射到不同的向量空间中，因此$b_{ij}$的更新为：
$$
b_{ij} = b_{ij} +\vec{u}^T_j S \vec{e}_i,i ∈ Iu , j ∈ {1, ..., K },
$$
$I_u$为用户历史行为，即$e_i$表示行为条目的嵌入，$u_j$表示兴趣胶囊，双线性映射矩阵$S∈R^{d×d}$在每对行为胶囊和胶囊之间共享。**S为共享矩阵**

2）为了适应第一个改变，胶囊网络中权重的初始化由全部设置为0变为基于正太分布的初始化。

![](https://i.loli.net/2020/11/25/3uV86IvSJmsYa5Q.png)

通过Multi-Interest Extractor Layer，得到了多个用户向量表示。接下来，每个向量与用户画像embedding进行拼接，经过两层全连接层（激活函数为Relu）得到多个用户兴趣向量表示。每个兴趣向量表征用户某一方面的兴趣。





##  Label-aware Attention Layer

计算每个兴趣胶囊与嵌入的目标项目之间的相容性，并计算兴趣胶囊的加权和作为目标项目的用户表示向量：

![](https://i.loli.net/2020/11/23/XOg6PLJKjYz45m1.png)

Q相当于目标商品的embedding，K和V都是用户的兴趣向量。值得注意的一点是，在softmax的时候，对得到的attention score，通过指数函数进行了一定的缩放。p是用于调整注意力分布的可调参数。当p接近0时（这里应该是假设了向量的内积大于1吧），对softmax是一种平滑作用，使得各attention score大小相近，当p>1且不断增加时，对softmax起到的是一种sharpen作用，attention score最大那个兴趣向量，在softmax之后对应的权重越来越大，此时更类似于一种hard attention，即直接选择attention score最大的那个向量。实验表明hard attention收敛速度更快。
$$
\vec{v}_u= Attention(\vec{e}_i,V_u,V_u) = V_u softmax(pow(V^T_u\vec{e}_i),p)
$$

##  Training & Serving

**训练时额外加入Label-aware Attention 来指导训练，但实际服务不是**

训练阶段，使用经过Label-aware Attention Layer得到的用户向量和目标商品embedding，计算用户u和商品i交互的概率：
$$
Pr(i|u) = Pr(\vec{e}_i | \vec{v}_u) = \frac{exp(\vec {v}^T_u\vec{e}_i)}{sum_{j \in I}exp(\vec{v}^T_u\vec{e}_i)}
$$
此处的 I 应该为用户u历史交互过的商品集合，即待推荐商品的分数与历史商品分数的比例最小

而目标函数（而非损失函数）为：
$$
L = \sum_{(u,i)\in D} logPr(i|u)
$$
而在线上应用阶段，只需要计算用户的多个兴趣向量，然后每个兴趣向量通过最近邻方法（如局部敏感哈希LSH）来得到最相似的候选商品集合。同时，当用户产生了一个新的交互行为，MIND也是可以实时响应得到用户新的兴趣向量的。





## EXPERIMENTS

### dataset

![截屏2020-11-26 下午3.24.54](https://i.loli.net/2020/11/26/W63UwcjnTdBfKCQ.png)

### Offline result

通过预测用户的下一次交互来评估方法性能，Hit rate被计算为：
$$
HitRate@N = \frac{\sum (u,i)∈D_{test} I(target \ item \ occurs \ in \ top\  N) }{|Dtest |}
$$
测试集的item出现在推荐列表的数量



$D_{test}$表示由用户对和目标项(u,i)组成的测试集，I 表示指标函数。

![截屏2020-11-26 下午3.37.34](https://i.loli.net/2020/11/26/Jb6UKRyMVT3C7WX.png)

1)多兴趣提取层利用聚类过程生成兴趣表示，实现更精确的用户表示。

2)标签感知注意层使目标物品参与多个用户表示向量，使用户兴趣与目标物品的匹配更加准确。



### 超参数分析



![截屏2020-11-26 下午4.05.07](https://i.loli.net/2020/11/26/ndiYvDF1P79tsze.png)



### Online EXPERIMENTS

![截屏2020-11-26 下午3.52.09](https://i.loli.net/2020/11/26/xGwBQR4FuhdCmaW.png)

1) 基于item的CF经过长期的实践优化，比YouTube的DNN表现更好，同样被MIND with single interest超越。

2) 一个非常明显的趋势是，当被提取的兴趣从1个增加到5个时，MIND的表现就会变得更好。

3) 7个兴趣数提升的是可以忽略的

4）动态兴趣数机制并没有带来CTR增益，但在实验中我们发现该方案可以降低服务成本，有利于天猫等大型服务，在实际应用中更具可接受性

### Coupling Coefficients



![截屏2020-11-26 下午4.07.44](https://i.loli.net/2020/11/26/1wu9Jn5SVtLYZAf.png)

从Tmall随机挑选两个用户，每行对应一个兴趣胶囊，每列为对应一个行为

由此可见，用户C(上)与4类商品(耳机、零食、手袋、服装)进行了交互，每一类商品在一个利益胶囊上的耦合系数最大，形成了相应的兴趣。

而用户D (下)只对衣服感兴趣，因此从行为上解决了3个粒度更细的兴趣(毛衣、大衣、羽绒服)。

根据这一结果，我们确认将每一类用户行为聚类在一起，形成相应的兴趣表示向量

### Item Distribution

![截屏2020-11-26 下午4.15.01](https://i.loli.net/2020/11/26/TR6eKgYuvaUdhyB.png)

与左侧示例的用户行为相对应的每个兴趣所回忆的项目分布。每个兴趣用一个轴表示，其中的坐标是项目和兴趣之间的相似性。点的大小与具有特定相似性的物品的数量成正比。

在服务时，类似于用户兴趣的项目是通过最近邻搜索来检索的。我们根据每个兴趣与相应兴趣的相似性来想象这些物品的分布。

图6显示了图5(上)提到的同一用户(用户C)的项分布。

MIND召回的物品与相应的兴趣高度相关，但YouTube DNN所召回到的项目随着项目类别的不同而差异很大，与用户行为的相似性较低。



收获：

1⃣️姿态矩阵W是可以共享的

2⃣️论文在训练时加入目标商品参与多个用户表示向量，使用户兴趣与目标物品的匹配更加准确，得到一个向量，但在上线服务时只是使用多个向量表示，根据不同兴趣找到对应商品，从而来快速找到n个候选商品。

论文侧重于快速召回，新闻推荐更侧重于更准确的用户特征的表示