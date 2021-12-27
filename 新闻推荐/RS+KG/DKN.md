# DKN: Deep Knowledge-Aware Networkfor News Recommendation

Hongwei Wang Shanghai Jiao Tong University & Microsoft Research Asia, Shanghai, China

 Fuzheng Zhang Microsoft Research Asia, Beijing,China

Xing Xie Microsoft Research Asia, Beijing

Jianlong Tan Chinese Academy of Sciences

Minyi Guo Shanghai Jiao Tong University, Shanghai, China

**Publication:** WWW '18: Proceedings of the 2018 World Wide Web Conference April 2018 Pages 1835–1844https://doi.org/10.1145/3178876.3186175

被引用：182

## 一、导读

本文提出了一种将知识图谱表示引入到新闻推荐中的深度知识感知网络(deep knowledge-aware network ，DKN)中

DKN是一个基于内容的深度推荐框架，用于点击率预测。DKN的关键部分是一个多通道和单词-实体对齐的知识感知卷积神经网络(KCNN)，它融合了新闻的语义层面和知识层面的表示。KCNN将单词和实体视为多个通道，并在卷积过程中显式地保持它们之间的对齐关系。

为了解决用户不同兴趣的问题，作者还在DKN中设计了一个注意力模块，以便动态地聚合当前候选新闻的用户历史记录。



**新闻推荐的挑战：**

1. 新闻文章具有高度的时效性，相关性在短时间内很快就会失效
2. 人们阅读新闻时对标题十分敏感，如何根据当前候选新闻多样化的阅读历史动态衡量用户的兴趣，是新闻推荐系统的关键

3. 新闻语言通常是高度浓缩的，由大量的知识实体和共识组成。但用户感兴趣的新闻不一定和他阅读的前一条新闻的实体有关。传统的语义模型或主题模型只能根据词的共现或聚类结构来判断它们之间的关联性，而很难发现它们之间潜在的知识层次的联系。因此，用户的阅读模式将被缩小到一个有限的范围，不能在现有推荐方法的基础上进行合理的扩展。

为了解决上述问题，作者将知识图谱引入新闻推荐系统中，提出了DKN模型。DKN是一种基于内容的点击率预测模型，它以一条候选新闻和一个用户的点击历史为输入，输出用户点击新闻的概率。



**具体实现：**

对于输入新闻，作者通过将新闻内容中的每一个词与知识图中的相关实体相关联来丰富其信息，还搜索并使用每个实体的上下文实体集(即知识图中的近邻)来提供更多的互补和可区分的信息。

设计KCNN将新闻的词级和知识级表示形式融合，生成知识感知的嵌入矢量



## 二、PRELIMINARIES（准备工作）

### Knowledge Graph Embedding

一个典型的知识图谱由数以百万计的实体-关系-实体三元组(h，r，t)组成，其中h、r和t分别表示三元组的头、关系和尾

**translation-based knowledge graphembedding method**：

TransE：如果(h,r,t）存在三元组关系，则假定 h+r 约等于 t，此处为向量。

![image-20200918203855379](/Users/mac/Library/Application Support/typora-user-images/image-20200918203855379.png)

评分函数越小，则网络中h,t的三元组关系（(h,r,t））越可靠。

TransH：通过将实体嵌入到关系超平面中，允许实体在不同的关系中有不同的表示。![image-20200918204046797](https://i.loli.net/2020/09/18/rjUCghGEQKtokpl.png)

TransR：为每个关系r引入一个投影矩阵Mr，将实体嵌入到相应的关系空间

TransD：将transr中的投影矩阵替换为实体关系对的两个投影向量的乘积



### CNN for Sentence Representation Learning

本文作者利用了一种经典的CNN结构，Kim CNN，来提取句子特征表示。![image-20200918204649475](https://i.loli.net/2020/10/21/coezil3B6LSUMFb.png)

用词向量表示句子：

![image-20200918210102739](/Users/mac/Library/Application Support/typora-user-images/image-20200918210102739.png)

经过一层卷积计算：

​                                   ![image-20200918210140066](https://i.loli.net/2020/09/18/CSEOgtFYPkQGDUp.png)

得到特征映射：

​                           ![image-20200918210207331](https://i.loli.net/2020/09/18/UuEl41xP9OTbdMn.png)

在特征映射c上使用max-over-time的pooling操作来识别最有意义的特征：

![image-20200918210249280](https://i.loli.net/2020/09/18/dI13yT7uPaGWlih.png)

此处图中利用多个滤波器(具有不同的窗口大小)来获得多个特征c，并将这些特征连接在一起得到最后的语句表示。

### PROBLEM FORMULATION

给定用户的单击历史以及新闻标题中的单词与知识图谱中的实体之间的关系，我们要预测的是：对于一个用户i，是否会点击他没有浏览过的候选新闻tj。

## 三、DKN

以一条候选新闻和一条用户点击的新闻历史作为输入，每条新闻都使用一个专门设计的KNCC来处理其标题并生成embedding。通过KNCC，可以得到了一组用户点击历史的embedding，为了获得用户对当前候选新闻的最终embedding，作者使用一种基于注意力的方法，自动地将候选新闻匹配到其点击的每一条新闻p，并用不同的权重来聚合用户的历史兴趣。最后将候选新闻与的embedding用户嵌入的新闻连接并输入到深度神经网络(DNN)中，计算用户点击候选新闻的预测概率

![image-20200918212529185](https://i.loli.net/2020/09/18/lun8fxFCLEQHwgP.png)



### Knowledge Distillation（知识提取，非知识蒸馏）

由Knowledge Distillation得到的实体embedding作为KCNN的输入。

![image-20200918215254116](https://i.loli.net/2020/10/21/Nq6598nlDZShzRF.png)

1）首先，为了区分新闻内容中的知识实体，作者利用实体链接技术来消除文本中提到的歧义，将它们与知识图中的预定义实体关联起来；

2）基于这些被识别的实体，构造了一个子图，并从原始知识图中提取出它们之间的所有关系链接。之后，将知识子图扩展到所有的实体中。注意，被识别的实体之间的关系可能很稀疏，而且缺乏多样性。

  3）构建好知识子图以后，可以利用前文所述多种知识图谱嵌入方法，如TransH, TansR,TransD来学习实体的向量表示；

4）将学到的实体表示作为KCNN的输入。

虽然最目前的知识图谱嵌入方法一般可以保留原图中的结构信息，但是在后续的推荐中使用单个实体的学习嵌入信息仍然是有限的。为了帮助识别实体在知识图中的位置，作者为每个实体提取额外的上下文信息。在知识图中，实体e的“上下文”被看作是其近邻的集合：

![image-20200918220656543](/Users/mac/Library/Application Support/typora-user-images/image-20200918220656543.png)

其中，r是关系，G是构建的知识图谱子图。由于上下文实体在语义和逻辑上通常与当前实体密切相关，上下文的使用可以提供更多的补充信息，并有助于提高实体的可识别性。给定实体e的上下文，上下文嵌入表示由其上下文实体的平均值计算得到:



![image-20200918220715713](https://i.loli.net/2020/09/18/bQiFqoZW3ePfzVH.png)

图5展示了一个上下文示例。

![image-20200918215311129](https://i.loli.net/2020/09/18/2PtLdwWnfCEuFm4.png)



### Knowledge-aware CNN

通过预训练模型可以得到新闻标题词嵌入w，经过Knowledge Distillationwi可以联系到一个实体embedding 和上下文embedding

组合单词和相关实体的一个简单方法是将这些实体视为“伪词”，并将它们连接到单词序列中。比如：![image-20200919144748323](/Users/mac/Library/Application Support/typora-user-images/image-20200919144748323.png)

但作者认为这样有一定的局限性：
1)连接策略打破了单词和关联实体之间的联系，不知道它们的对齐方式

2)词嵌入和实体嵌入是由不同的方法学习,它们可能并不适合融合在单个向量空间中

3)连接策略隐式地强制词嵌入和实体嵌入具有相同的维数，这在实际设置中可能不是最优的，因为词嵌入和实体嵌入的最优维数可能不同

因此作者提出KCNN，将词汇语义和知识信息相结合：

由新闻标题得到词嵌入和应的实体向量表示和上下文向量表示，分别经过转换函数得到：

![image-20200919145519193](/Users/mac/Library/Application Support/typora-user-images/image-20200919145519193.png)

![image-20200919145605642](/Users/mac/Library/Application Support/typora-user-images/image-20200919145605642.png)

其中g是映射函数，可以是线性的，也可以非线性：

![image-20200919145641710](/Users/mac/Library/Application Support/typora-user-images/image-20200919145641710.png)

![image-20200919152543506](/Users/mac/Library/Application Support/typora-user-images/image-20200919152543506.png)

将三种embedding矩阵对齐叠加可以得到多通道的词表示W：

![image-20200919152527439](/Users/mac/Library/Application Support/typora-user-images/image-20200919152527439.png)

利用形如KCNN中不同窗口大小的滤波器对W进行处理获取新闻标题的表示：

![image-20200919152731159](/Users/mac/Library/Application Support/typora-user-images/image-20200919152731159.png)

使用一个max- overtime池操作来选择最大的特性：

![image-20200919152817750](/Users/mac/Library/Application Support/typora-user-images/image-20200919152817750.png)

最后组合为新闻标题的最后嵌入：

![image-20200919152900191](/Users/mac/Library/Application Support/typora-user-images/image-20200919152900191.png)

### Attention-based User Interest Extraction

建立一个attention network来建模用户点击的不同新闻对候选新闻的影响。

以用户单条点击新闻为例，首先将其嵌入和候选新闻的嵌入连接起来，然后用一个DNN作为注意网络和Softmax函数来计算归一化的影响力权重：![image-20200919153644137](https://i.loli.net/2020/09/19/MdUyvtON2xaR6ZS.png)

DNN注意力网络H接收两个新闻标题的嵌入作为输入并输出影响权重。因此，用户i相对于候选新闻tj的表达可以计算为他点击的新闻标题嵌入表示的加权和：

![image-20200919153808730](/Users/mac/Library/Application Support/typora-user-images/image-20200919153808730.png)

此时，已经得到候选新闻的embedding 和 用户的embedding，使用另一个DNN来计算用户i点击新闻tj的概率：
 ![image-20200919155140563](/Users/mac/Library/Application Support/typora-user-images/image-20200919155140563.png)

## 四、EXPERIMENTS

数据来源于Bing News.的服务器日志，以2016年10月16日至2017年6月11日的随机采样平衡数据集为训练集，2017年6月12日至2017年8月11日的随机采样平衡数据集为测试集

在Microsoft Satori知识图谱中搜索数据表中所有发生的实体以及它们单跳内的实体，并以大于0.8的置信度提取它们之间的所有边(三组)![image-20200919160834612](/Users/mac/Library/Application Support/typora-user-images/image-20200919160834612.png)

### Baselines

1. LibFM

2. KPCNN

3. DSSM

4. DeepWide

5. DeepFM

6. YouTubeNet

7. DMF

除了LibFM之外，所有的模型都是基于深度神经网络的。

除了DMF，其他都是基于内容的或混合方法

### Results

以F1和AUC为评价指标，选择TransD转换函数处理知识图和学习实体的嵌入，KCNN中并使用非线性变换函数。

嵌入维度为100，对于每个大小为1、2、3、4的窗口，过滤器的数量被设置为100。我们使用Adam[21]对DKN进行训练，优化log loss，每个实验重复五次，我们报告平均偏差和最大偏差作为结果。

#### 与baseline对比的实验结果：

![image-20200919162441462](https://i.loli.net/2020/10/21/RIh45ruwUTl6KcF.png)

DMF效果最差表明协同过滤不适用于新闻推荐

除了DMF，LibFM效果最差表明深度神经网络可以有效地捕获新闻数据中的非线性关系和依赖关系

DKN变体的比较：![image-20200919162644334](/Users/mac/Library/Application Support/typora-user-images/image-20200919162644334.png)

奇怪的是，DKN的各种变体的数据和其在上面与基于基线的对比数据有所不同，且不同变体的最佳效果的数据也不一致。

#### Analyze

随机抽取了一个用户，并从训练集和测试集中提取他的所有日志，并手动标记新闻的category

![image-20200919164108055](/Users/mac/Library/Application Support/typora-user-images/image-20200919164108055.png)

作者使用全部的训练数据来训练DKN的全特征和没有实体或上下文嵌入的DKN，然后将该用户的每一对可能的训练日志和测试日志提供给这两个经过训练的模型，并获得它们的注意网络的输出值。结果如图所示，较深的蓝色表示更大的注意力值。

![image-20200919164318381](/Users/mac/Library/Application Support/typora-user-images/image-20200919164318381.png)

包含知识图的DKN(图8b)准确地预测了所有的测试日志，而没有知识图的DKN(图8a)在第三个测试日志中失败

## 总结

文章的特点在于使用知识图来提取新闻之间的潜在知识层面的联系，以便在新闻推荐中更好地探索，相比其他baseline模型，KCN专攻于新闻推荐。

深度推荐系统可以分为两类:一类是利用深度神经网络对用户或物品的原始特征进行处理，另一类是利用深度神经网络对用户和物品之间的交互进行建模