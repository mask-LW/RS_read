# Neural News Recommendation with Attentive Multi-View Learning

Chuhan Wu, Tsinghua University, Beijing, China 

Fangzhao Wu, Microsoft Research Asia, Beijing, China

Mingxiao An, University of Science and Technology of China, Hefei, China 

Jianqiang, Huang Peking University, Beijing, China

Yongfeng Huang, Tsinghua University, Beijing, China 

 Xing Xie, Microsoft Research Asia, Beijing, China

Publication: Proceedings of the Twenty-Eighth International Joint Conference on Artificial Intelligence (IJCAI-19)

被引用：4

## 一、主要工作

基于新闻标题学习新闻表示是不够的，因此作者提出通过利用不同种类的新闻信息学习用户和新闻的特征表示，主要由news coder和user coder组成。

news coder：提出attentive multi-view learning model （专注的多视图学习模型）将标题、正文和主题类别联合起来学习新闻的特征表示。此外还同时word-level 和view-level 的attention mechanism选择信息丰富的word和view来学习新闻表征。

user coder：基于用户浏览的新闻，使用注意力机制选择信息丰富的新闻学习用户的特征表示。

## 二、研究背景

基于深度学习的新闻推荐算法的核心在于如何学习用户和新闻的表征。

**研究驱动**：

1.新闻文章通常包含不同种类的信息，例如标题，正文和主题类别，这些信息对于表示新闻都是有用的

2.不同种类的新闻信息具有不同的特征

3.不同的新闻信息可能对不同的新闻具有不同的信息性

## 三、NAML

![image-20200924214952892](https://i.loli.net/2020/10/21/epuHU4yomIY5j7x.png)



### News Encoder

![image-20200924102142791](https://i.loli.net/2020/10/21/tD2gnfEZlNiTHsq.png)

News Encoder主要由四个附件组成：

1.title encoder

（1）Word Embedding：由新闻标题的序列得到对应的词嵌入向量，由w得到e

（2）CNN：捕捉对应新闻标题表征重要的上下文表示，输出是一系列上下文词表征

![image-20200924102406155](file:///Users/mac/Library/Application%20Support/typora-user-images/image-20200924102406155.png?lastModify=1600914848)

（3）word-level attention network：使用词级注意力网络在每个新闻标题的上下文中选择重要的词

该网络参考[Wuet al., 2019b]Chuhan  Wu,  Fangzhao  Wu,  Junxin  Liu,Shaojian He, Yongfeng Huang, and Xing Xie.  Neural de-mographic prediction using search query. InWSDM, pages654–662. ACM, 2019.

词权重计算：

![截屏2020-10-12 下午9.04.17](https://i.loli.net/2020/10/12/voHAthc3l2UE58R.png)

新闻标题的最后表示是其词的上下文通过他们的注意权重加权求和来进行表示

![image-20200924103354957](file:///Users/mac/Library/Application%20Support/typora-user-images/image-20200924103354957.png?lastModify=1600914848)

2.body encoder：类似于title encoder

3.category encoder：新闻文章通常都贴上主题分类的标签，甚至副标签，比如sport对应子标签basketball等

【使用该方法需要新闻有类别标签】

以标签和子标签的ID作为输入，经过：

（1）Word Embedding：得到对应的低维稠密的表征

（2）Dense层：![image-20200924211734821](/Users/mac/Library/Application Support/typora-user-images/image-20200924211734821.png)



4.attentive pooling：不同种类的新闻信息可能具有不同的信息量，以便学习不同新闻的表征

有的新闻标题包含足够多信息，其标题应对于新闻表征具有较高权重；有的新闻标题没有包含足够多信息，则应该在其正文和类别应该比title有更高的权重来代表这个新闻。

提出了一个view-level的attention网络，对不同类型新闻信息的信息量进行建模，以学习新闻表征

标题权重计算：

![image-20200924212329342](/Users/mac/Library/Application Support/typora-user-images/image-20200924212329342.png)

正文、类别、子类别的计算类似



最后得到的新闻表征为不同观点新闻表征的总和：

![image-20200924212523100](/Users/mac/Library/Application Support/typora-user-images/image-20200924212523100.png)

**News Encoder用于获取用户浏览的历史新闻和要推荐的候选新闻的表示**

### User Encoder

同一用户浏览的不同新闻具有不同的代表该用户的信息，使用attention网络，通过选择重要的新闻来了解更多的信息。



用户浏览的第i条新闻表示为：

![image-20200924214621768](/Users/mac/Library/Application Support/typora-user-images/image-20200924214621768.png)

r表示由News Encoder得到的新闻表征

用户的最终表征是用户浏览的新闻的表现形式的总和，并根据用户的注意力权重进行加权：

![image-20200924214702328](/Users/mac/Library/Application Support/typora-user-images/image-20200924214702328.png)

### Click Predictor

作者认为使用内积来计算 click  probability  score是最有效率，性能最佳

![image-20200924215413262](/Users/mac/Library/Application Support/typora-user-images/image-20200924215413262.png)

### Model Training

使用负采样进行模型训练。

对于一个用户浏览过的每条新闻，我们将其视为正样本，随机抽取在同一会话中预先发布但未被该用户点击的新闻作为负样本，然后共同预测一个正样本和负样本的点击概率的分数，将新闻点击预测问题构建为一个伪k +1路分类任务。

正样本：![image-20200924220344917](/Users/mac/Library/Application Support/typora-user-images/image-20200924220344917.png)

负样本：![image-20200924220415604](/Users/mac/Library/Application Support/typora-user-images/image-20200924220415604.png)

使用softmax对这些点击概率分数进行归一化，然后计算一个正样本的后验点击概率：

![image-20200924220333655](/Users/mac/Library/Application Support/typora-user-images/image-20200924220333655.png)

$$
y^{^+}_i表示第i个正样本的分数，y^{^-}_{i,j}表示第i个正样本对应session下第j个负样本的分数，
$$
loss function：

![image-20200924220517575](/Users/mac/Library/Application Support/typora-user-images/image-20200924220517575.png)

S是正训练样本的集合。

## 四、Experiments

我们的实验是在一个真实数据集上进行的，该数据集是随机抽取一个月内msn news的用户日志，即2018年12月13日至2019年1月12日，使用最后一周的日志作为测试集，其余的日志用于训练集，随机抽取训练样本的10%作为验证集。

![截屏2020-09-25 下午7.59.26](/Users/mac/Library/Application Support/typora-user-images/截屏2020-09-25 下午7.59.26.png)

根据验证集选择参数：

词嵌入维度：300

category嵌入维度：100

预训练模型：Glove

CNN过滤器的数量为400个，窗口大小为3个

category视图中密集层的维度被设置为400。注意力查询的大小被设置为200

负采样率K：4

对每一层应用了20%的dropout (Srivastavaet al.，2014)来减轻过拟合

使用Adam优化算法，批处理大小为100

实验中的指标包括所有展示的平均AUC，MRR，nDCG @ 5和nDCG @ 10得分。

分别将每个实验重复10次，并报告了平均效果。

### result

![截屏2020-09-25 下午8.34.04](/Users/mac/Library/Application Support/typora-user-images/截屏2020-09-25 下午8.34.04.png)

### Effectiveness of Attentive Multi-View Learning（专注多视角学习的效果）

![截屏2020-09-25 下午9.17.36](/Users/mac/Library/Application Support/typora-user-images/截屏2020-09-25 下午9.17.36.png)

由Figure3（a）得：

body view提供的信息比title/category更丰富，结合这三个view可以进一步提高我们的方法的性能





















