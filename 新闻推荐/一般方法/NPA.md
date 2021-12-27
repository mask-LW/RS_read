



# NPA: Neural News Recommendation with Personalized Attention

Chuhan Wu, Tsinghua University, Beijing, China 

Fangzhao Wu, Microsoft Research Asia, Beijing, China

Mingxiao An, University of Science and Technology of China, Hefei, China 

Jianqiang, Huang Peking University, Beijing, China

Yongfeng Huang, Tsinghua University, Beijing, China 

 Xing Xie, Microsoft Research Asia, Beijing, China

**Publication:** KDD '19: Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining July 2019 Pages 2576–2584 https://doi.org/10.1145/3292500.3330665

Cited times:12

## 一、研究背景

不同的用户通常有不同的兴趣，同样的用户可能有不同的兴趣。因此，不同的用户可能会在点击同一篇新闻时关注不同的方面

![截屏2020-09-26 下午8.49.56](https://i.loli.net/2020/10/23/4HxpmKJkntf9weq.png)



1.并非用户点击的所有新闻都能反映用户的偏好。 
2.相同的新闻还应该具有不同的信息量，以便为不同的用户建模。 
3.新闻标题中的不同单词通常对于学习新闻表示形式具有不同的信息性。 
4.新闻标题中的相同单词在揭示不同用户的偏爱方面也可能具有不同的信息性。 



为不同用户建模单词和新闻的不同信息性可以有助于学习更好的新闻特征表示和用户表示以进行准确的新闻推荐

## 二、NPA

![image-20200920105922745](https://i.loli.net/2020/10/21/gPABZJoSCLGh2Ir.png)

### News Encoder：基于新闻标题学习新闻文章的特征表示

**重点：词汇层面的个性化注意网络（word-level personalized attention network）帮助NPA为不同用户选择重要词汇 **

首先从新闻的标题序列得到词嵌入向量E，将其喂入CNN中，用以捕捉新闻标题的局部信息，得到
$$
c_i = ReLU(F_w * e_{(i-k):(i+k)} + b_w)
$$
最后经过word-level personalized attention network

1⃣️新闻标题的单词有些可以提供重要信息，有些不可以，比如代词，

2⃣️不同用户对于新闻标题的信息的感兴趣程度是不同的，比如，车辆信息为对车辆不感兴趣的用户所提供的信息是很少的，但为对车辆感兴趣的用户提供丰富的信息。

因此还需要一个word-level personalized attention network，根据用户的喜好来识别和强调重要的词。

![image-20200921104235320](https://i.loli.net/2020/10/23/oPmD7UstpW2C9Iw.png)

首先将用户ID嵌入表示，通过一个Dense层得到一个用户偏好query qw，然后通过偏好query和单词嵌入的交互，得到每个单词的注意力权重α，最后第i个新闻标题的表示为输入的加权求和。
$$
q_w = ReLU(V_w * e_u + v_w),(2)
$$

$$
a_i = c^T_itan_h(W_p * q_w + b_p),(3)
$$

$$
\alpha_i = \frac{exp(a_i)}{\sum^M_{j=1}exp(aj)},(4)
$$

$$
r_i = \sum^M_{j=1}\alpha_jc_j,(5)
$$

**News Encoder分别得到每个用户已点击新闻的特征表示和每个候选新闻的特征表示**

### User Encoder：从用户已经点击的新闻得到用户的表示（这里不是偏好）

重点：新闻级的个性化关注网络（news-level personalized attention）帮助NPA为不同用户选择重要的历史点击新闻

同一用户点击的不同新闻为用户提供的信息是不同的。

同样的新闻对应不同的用户所提供的信息也是不同的。

在News Encoder的基础上为了建模同样新闻对不同用户所提供的不同信息，同样采用personalized attention mechanism,计算出不同新闻对应的权重，得到一个用户的特征表示u:


$$
q_d = ReLU(V_d * e_u+v_d),(6)
$$

$$
a'_i = r_i^Ttanh(W_d * q_d + b_d),(7)
$$

$$
\alpha'_i = \frac{exp(a'_i)}{\sum_{j=1}^Nexp(a'_j)},(8)
$$

$$
u = \sum^N_{j=1}\alpha'_jr_j,(9)
$$


News Encoder针对一个标题的不同单词，Uers Encoder针对一个用户点击的不同新闻。



### Click Predictor

Click Predictor是用来计算一个用户点击不同的候选新闻的分数

新闻推荐中的一个常见观察结果是，大多数用户通常只点击展示中显示的一些新闻。因此，正面和负面新闻样本的数量高度不平衡。许多方法通过随机采样手动平衡正负新闻样本，从而丢失了负样本提供的丰富信息。（直接使用sigmoid function）

此外，由于新闻样本的总数通常很大，因此在模型训练期间这些方法的计算成本通常很高。因此，这些方法在模拟现实世界中的新闻推荐场景时不是最佳选择。

作者建议通过联合预测模型训练期间K + 1个新闻的点击得分来应用负采样技术，以解决上述两个问题。

 K + 1个新闻由一个用户的正样本和一个用户随机选择的负样本组成

首先通过新闻和用户表示向量的内积来计算候选新闻的分数ˆyi，然后通过softmax函数对其进行归一化，然后进行模型训练，将其视为K+1的分类问题，即点击新闻为正类，其余所有新闻为负类。 我们应用最大似然法来最小化正类的对数似然:
$$
\hat{y}^{'}_i =  r^{'T}_iu,(10)
$$

$$
\hat{y}^{'}_i = \frac{exp(\hat{y}^{'}_i)}{\sum^K_{j=0}exp(\hat{y}^{'}_i)},(11)
$$

$$
L = -\sum_{y_j \in S} log(\hat{y}_j),(12)
$$

与现有的新闻推荐方法相比，我们的方法可以有效地挖掘负面样本中的有用信息，并进一步降低模型训练的计算成本（几乎除以K），因此可以在大量新闻点击日志上更轻松地训练模型 。

## 三、EXPERIMENTS

数据集是通过一个月（即2018年12月13日至2019年1月12日）对MSN News的用户日志进行随机抽样而构建的。![image-20200921171434565](https://i.loli.net/2020/10/23/MDOovLuf8ElYUXr.png)

最后一周作为测试集，其余为训练集，随机采样10%作为验证。

实验中的指标包括所有展示的平均AUC，MRR，nDCG @ 5和nDCG @ 10得分。

分别将每个实验重复10次，并报告了平均效果。

### setting

word embedding维度：300

预训练模型：Glove

CNN过滤器Nf的数量设置为400，窗口大小为3

用户embedding的大小：50

word和news偏好查询的大小：200

负采样率K：4

使用dropout缓解过拟合，rate=0.2

使用Adam优化算法，批处理大小为100

由于GPU内存的限制，在基于神经网络的方法中，用于学习用户表示的单击新闻的最大数量设置为50，新闻标题的最大长度设置为30。

### baseline

1. LibFM 
2. CNN
3. DSSM
4. DeepWide（2016）：结合前馈神经网络(深)和线性模型(宽)进行推荐的基于深度学习的模型
5. DeepFM（2017）：一种基于内容的深度神经网络
6. DFM（2017）：典型的基于cf的新闻推荐模型充分利用了显性评分和隐性反馈
7. DKN（2018）：深度推荐框架，设计了一个多频道CNN，融合了新闻的语义层和知识层表示，并引入了预测点击率的注意机制

result

![image-20200921172507698](https://i.loli.net/2020/10/23/LJYF3mgI9CEv6cA.png)

1. 基于神经网络的方法要比传统的方法如LibFM要好
2. 使用负采样的DSSM和NPA比其他方法更好
3. 使用注意力机制的方法如DFM、DKN和NPA比CNN、DSSM、Wide&Deep、DeepFM（没有使用注意力机制）要好
4. NPA最佳

![image-20200921173205884](https://i.loli.net/2020/10/23/ysSWQexBCNziTwp.png)

###  Effectiveness of Personalized Attention

![image-20200921203724074](/Users/mac/Library/Application Support/typora-user-images/image-20200921203724074.png)

![image-20200921203738814](/Users/mac/Library/Application Support/typora-user-images/image-20200921203738814.png)

### Influence of Negative Sampling

![image-20200921204314087](/Users/mac/Library/Application Support/typora-user-images/image-20200921204314087.png)

![image-20200921204327085](/Users/mac/Library/Application Support/typora-user-images/image-20200921204327085.png)

由于负样本在训练集中占主导地位，它们通常包含丰富的推荐信息，但是由于平衡样本的局限性，负样本所提供的信息无法得到有效利用，因此性能通常不是最佳的。采用负样本取样可以更好地利用负样本所提供的信息。

###  Case Study

通过实例以可视化方式查看个性化注意力机制的有效性。

![image-20200921205350592](/Users/mac/Library/Application Support/typora-user-images/image-20200921205350592.png)

可以看到，对于一些更加重要的词所对应的权重更大。



![image-20200921205401146](/Users/mac/Library/Application Support/typora-user-images/image-20200921205401146.png)

此外，根据不同用户的偏好，为相同新闻标题中的单词选择不同的关注权重。

例如（a），根据点击的新闻，用户1可能对体育新闻感兴趣，而用户2可能对电影相关新闻感兴趣。 为用户1高亮显示单词“超级碗”，为用户2高亮显示单词“超级英雄”和“电影”。

例如（b）,新闻“美国国家橄榄球联盟发布季后赛时间表：周六晚上的牛仔-海鹰”获得了很高的关注权重，因为这对于建模用户1的偏好非常有用，因为他/她很可能会根据点击的新闻对体育感兴趣,但为用户2分配了相对较低的关注权重.



*待看*



1⃣️CNN是如何捕捉局部信息。

已解决计算过程。

2⃣️传统attention network如何操作

3⃣️个性化待attention network如何通过用户ID得到偏好

看代码怎么解决。

4⃣️如何提出负采样技术来训练

5⃣️实验指标的意义