# Neural Graph Collaborative Filtering

这是何向南教授团队在SIGIR2019发表的一篇文章，主要针对隐式反馈行为数据。为了解决传统的方法主要通过user或item的pre-existing fetures的映射得到user和item的embedding，而缺乏了user和item之间重要的协同信息(collaborative signal)这一问题，作者提出了一种新的推荐框架——神经图协同过滤。

核心的目标问题在于如何更好的将user-item之间的协同信息建模到user和item的emebdding表示中。

传统的方法无法学到很好的Embedding，归咎于缺乏对于重要的**协同信息**显示地进行编码过程，这个协同信息隐藏在user-item交互数据中，蕴含着users或者items之间的行为相似性。更具体地来说，传统的方法主要是对ID或者属性进行编码得到embedding，再基于user-item interaction定义重构损失函数并进行解码。可以看出，user-item interaction只用到了解码端，而没有用到编码端。这样会导致，学习到的embedding的信息量不够，为了更好的预测，只能依赖于复杂的interaction (decoder) function来弥补次优的embedding在预测上的不足。

由于原始的interraction数据规模较大，难以将协同信号建模到embedding的表示当中。

作者提出在user-item interaction graph structure图结构中进行collaborative signal的编码。

# 前言

![截屏2021-05-21 下午4.28.24](https://i.loli.net/2021/05/21/GiqtxIg4mTwsNkp.png)

推荐的目标用户为u1，左边为用户-物品交互图，右边为从u1展开的树形结构。高阶连通性表示路径长度l大于1的任意节点到u1的路径。这种高阶连接包含了丰富的语义，这些语义承载着协同信号。

u1 ← i2 ← u2 表示了u1和u2之间的行为相似性，因为两个用户都与i2有交互;

u1 ← i2 ← u2 ← i4 表明u1很可能对i4感兴趣，因为他的相似用户已经选择i4。

从l=3的角度来看，u1对i4比i5更感兴趣，因为i4有两条路径通往u1







# 模型结构

![截屏2021-05-21 下午12.12.22](https://i.loli.net/2021/05/21/VkZF6UBEy1Pcaxz.png)

### Embedding Layer

初始化用户ID和物品iD的词嵌入：

![截屏2021-05-21 下午12.15.58](/Users/mac/Library/Application Support/typora-user-images/截屏2021-05-21 下午12.15.58.png)

传统的方法直接将E输入到交互层，进而输出预测值。而NGCF将E输入到多层嵌入传播层，通过在二部图上进行嵌入的传播来对嵌入进行精炼。

### Embedding Propagation Layers

定义GNN的消息传播框架，包括两个步骤，Message construction和Message aggregation。

##### **Message construction**

对于每个user-item pair(u,i)，定义从i到u传递的messege如下：

![截屏2021-05-21 下午12.19.30](https://i.loli.net/2021/05/21/RpzLGXTjOufAgWs.png)

$p_{ui}$来控制每条边edge (u,i)在每次传播中的衰减系数,f为message encoding function

作者使用的message encoding function：

![截屏2021-05-21 下午12.21.06](https://i.loli.net/2021/05/21/9cMTxhdHrg6Fkv4.png)

传统的图卷积只考虑消息的来源，此处还考虑了信息来源和信息目的地的关系，即$e_i \bigodot e_u$

element-wise product是典型的特征交互的一种方式

![截屏2021-05-21 下午12.26.33](https://i.loli.net/2021/05/21/cntp6BSj7LdzQuV.png)

从信息传递角度，$p_u$可以看做是折扣系数，随着传播路径长度的增大，信息慢慢衰减（这个可以通过叠加多层，并代入到式子，会发现前面系数也是连乘在一起了，说明路径越长，衰减越大）。

后面两个符号分别表示用户1-hop的邻居节点数和item1-hop的邻居节点数

##### **Message Aggregation**

将从用户u的邻居传递过来的信息进行汇聚来重新定义用户u的嵌入表示

![截屏2021-05-21 下午12.29.17](https://i.loli.net/2021/05/21/fMZBkYHLUOhmtAs.png)

LeakyReLU允许对于正反馈信息和少部分负反馈信息的编码

除了从邻居u传播的消息，我们取u的自连接以保持原有特征

![截屏2021-05-21 下午12.30.38](https://i.loli.net/2021/05/21/Stdn4QLIeGg6JZO.png)

此处与公式（2）共享权重W1

item的嵌入表示同理可得。

embedding propagation layer的好处在于显示地挖掘 first-order connectivity信息来联系user和item的表示。

以上作为下一个embedding propagation layer的输入，可以叠加多层，挖掘multi-hop的关联关系

![截屏2021-05-21 下午3.45.41](https://i.loli.net/2021/05/21/RhaHWqUvgwnBijb.png)

此时传递的信息定义为：

![截屏2021-05-21 下午3.46.30](https://i.loli.net/2021/05/21/upsMGjr5qFkWBT3.png)



![截屏2021-05-21 下午3.47.45](https://i.loli.net/2021/05/21/JCMVQf6pxclihSW.png)

如图所示，联合信息比如: i4->u2-> i2-> u1可以从传播过程中捕获

来自i4的信息可以通过红线一直传播至u1，编码在u1的表征里面

进一步，作者给出了上述Equation(5), (6) 过程的矩阵表示形式，有助于实现layer-wise propagation。

![截屏2021-05-21 下午3.51.46](https://i.loli.net/2021/05/21/EIoDibem3C8UcZS.png)

![截屏2021-05-21 下午4.01.53](https://i.loli.net/2021/05/21/PdqExs5oObhZkRJ.png)



其中 $E^(l) \in R^{(N+M)*d_l}$,是user，item共同进行传播的表示形式，即把user, item的embeddings矩阵concat在一起，一起进行传播



L为user-item interaction graph的拉普拉斯矩阵

![截屏2021-05-21 下午3.55.58](https://i.loli.net/2021/05/21/KzQhM9jrlxw6RDq.png)

**R**是user-item 交互矩阵，**0**是全0矩阵

A是邻接矩阵，是user-item 交互矩阵和item-user交互矩阵构成的

![截屏2021-05-21 下午4.03.03](https://i.loli.net/2021/05/21/Rrbf6pDsE3l7jhm.png)

D是对角度矩阵

### Model Prediction

最终的嵌入表示是原始的embedding和所有嵌入传播层得到的embedding全部concat在一起的结果。即：

![截屏2021-05-21 下午4.05.41](https://i.loli.net/2021/05/21/1SpRBkDHVqnmiTQ.png)

其中｜｜表示连接操作，0对应初始化嵌入

还是用点乘预测交互：
![截屏2021-05-21 下午4.07.54](/Users/mac/Library/Application Support/typora-user-images/截屏2021-05-21 下午4.07.54.png)

### Optimization

采用的是pairwise BPR loss进行优化：

![截屏2021-05-21 下午4.10.20](https://i.loli.net/2021/05/21/lHmugosftReJDjv.png)

![截屏2021-05-21 下午4.09.18](/Users/mac/Library/Application Support/typora-user-images/截屏2021-05-21 下午4.09.18.png)

O为成对训练数据，R+为观察到的相互作用，R−为未观察到的相互作用



![截屏2021-05-21 下午4.10.50](https://i.loli.net/2021/05/21/pYfeKEihnXHkTav.png)

此处是所有可训练参数

## NGCF框架应用于SVD++

SVD++可以看作是没有高阶传播层的NGCF的一个特例,也就是说仅有一层propagation layer,

禁用了转换矩阵和非线性激活函数：


![截屏2021-05-21 下午4.17.18](/Users/mac/Library/Application Support/typora-user-images/截屏2021-05-21 下午4.17.18.png)



# EXPERIMENTS

作者的实验从以几个方面来展开

- RQ1:与最先进的CF方法相比，NGCF的性能如何?
- RQ2:不同的超参数设置(如层深度、嵌入传播层、层聚合机制、dropout)如何影响NGCF?
- RQ3:表示如何从高阶连接中获益?

作者采用了Gowalla, Yelp2018, Amazon-book 这三个数据集





![截屏2021-05-21 下午4.19.51](https://i.loli.net/2021/05/21/MIwF1qiflpG86Yj.png)



### RQ3

从Gowalla数据集中随机选择了6个用户及其相关条目,观察它们的表现如何影响NGCF的深度。

![截屏2021-05-21 下午4.24.17](https://i.loli.net/2021/05/21/51ZXGhOQoN79gDt.png)

​	用户与物品的连接性在嵌入空间中得到了很好的体现，即嵌入到空间的邻近部分。特别是，NGCF-3呈现出可识别的聚类，这意味着具有相同颜色的点(即相同用户所消费的道具)倾向于形成聚类。

​	分析相同的用户,比如黄色用户,当叠加三个嵌入传播层时，它们的历史项的嵌入更接近,这定性的验证了NGCF的有效性,