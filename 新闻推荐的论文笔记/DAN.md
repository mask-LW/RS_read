# DAN: Deep Attention Neural Network for News Recommendation

**Qiannan Zhu** Chinese Academy of Sciences

**Xiaofei Zhou **Chinese Academy of Sciences

**Zeliang Song **Chinese Academy of Sciences

**Jianlong Tan **Chinese Academy of Sciences

**Li Guo **Chinese Academy of Sciences

**Publication：**July 2019 Proceedings of the AAAI Conference on Artificial Intelligence 33:5973-5980

被引用：20

## 一、

个性化新闻推荐的典型方法：

1. collabora-tive  filtering  (CF)  协同过滤
2. content  based  methods 基于内容的方法
3. hybrid methods

以上方法会出现冷启动问题

4.基于深度学习的方法：具有更好的对复杂的用户-项(即新闻)交互进行建模，捕捉新闻和用户的动态属性的能力

DKN实现了最先进的推荐性能。



在此之前深度学习做新闻推荐的问题：

1. 只考虑新闻标题作为推荐特征，而不考虑news Profile
2. 没有考虑用户点击新闻的顺序信息的影响



因此作者提出了一个deepattention  neural  network （ DAN）

学习新闻特征：PCNN组件，由两个并行卷积神经网络组成，用于融合news title 和 news Profile

学习用户特征：ANN建模用户当前兴趣，ARNN（attention-basedrecurrent neural network）捕捉用户阅读历史的顺序特征

DAN考虑 用户是否根据新闻特征和用户特征的相似度点击候选新闻，由PCNN、ARNN和ANN组成。



问题描述：
$$
假设已经得到一个用户的历史新闻点击列表\lbrace{x1,x2...x_t-1}\rbrace,x_j为候选新闻，推荐系统将用户的新闻阅读历史列表作为输入，计算用户点击x_j的概率
$$

## 二、DAN Framework

![image-20200917112648267](https://i.loli.net/2020/09/17/hmA3MWZtRsdE6BX.png)

（1）PCNN从用户浏览新闻历史，得到新闻的title-level 和 profile-level 的特征表示

（2）右边的ANN在（1）的基础上，将候选新闻与每条被点击的新闻进行匹配，并聚合到用户当前兴趣的特征表示，得到user interet embedding

（3）ARNN将注意力机制施加在RNN的每个重复模块上，以捕获不同点击时间的序列特征，得到user history sequential embedding

（4）将整个序列特征和用户当前兴趣特征组合连接到一个全连接的神经网络中，得到最终的用户特征表示。





### PCNN:News Representation Extractor

以新闻标题和新闻轮廓为输入，学习新闻标题级和新闻轮廓级的表示，二者组合为新闻的特征表示

![image-20200917152913840](https://i.loli.net/2020/09/17/37Ud5pHANGmQfqE.png)



### ANN:User Interest Extractor

根据用户点击的历史新闻获取用户对候选新闻的当前兴趣。

![image-20200917153442365](/Users/mac/Library/Application Support/typora-user-images/image-20200917153442365.png)

设计了一个基于注意力的神经网络，自动将每个点击的新闻与候选新闻进行匹配，并以不同的权重聚合用户当前的兴趣

已知用户浏览历史{x1,x2...xt-1},用{I1,I2,...It-1}表示其embedding，目标新闻用It表示

![image-20200917154534271](https://i.loli.net/2020/09/17/9cmUiv7IypxXT2N.png)

得到的St为用户的兴趣特征表示

### ARNN:Sequential Information Extractor

ARNN是基于attention的RNN，从用户的历史阅读中捕捉用户的阅读历史的序列特征，此处假设用户每次的点击都会受到之前新闻选择的影响

ARNN以用户浏览的新闻历史的embedding作为输入，输出用户的阅读历史顺序特征表示。

此处的ARNN的ANN和上面提到的ANN一样，但参数不同，通过ANN得到序列特征矩阵s，再经过integra-tion function将矩阵转化为向量h。此处有四种方法：

![image-20200917160116796](/Users/mac/Library/Application Support/typora-user-images/image-20200917160116796.png)

### Similarity

最后将h和St组合起来喂入一个全连接层，得到user embedding，通过余弦相似度计算user embedding和candidate news embedding的相似度得到概率：

![image-20200917160627548](/Users/mac/Library/Application Support/typora-user-images/image-20200917160627548.png)

## 三、Experiment

### train

用现有的观察到的点击历史阅读作为正样本，使用未观察到的历史阅读作为负样本来训练我们的DAN模型，

正样本对应标签为1，负样本对应标签为0。

数据经过模型后得到每个输入得到对应用户点击候选新闻的概率。

通过最小化负对数似然函数来训练我们的模型

![image-20200917163103604](https://i.loli.net/2020/09/17/WbyUOfXGZrigeYN.png)

### Baselines

1. LibFM 

2. DSSM
3. DeepWide（2016）：结合前馈神经网络(深)和线性模型(宽)进行推荐的基于深度学习的模型
4. DeepFM（2017）：一种基于内容的深度神经网络
5. DMF（2017）：典型的基于cf的新闻推荐模型充分利用了显性评分和隐性反馈
6. DKN（2018）：深度推荐框架，设计了一个多频道CNN，融合了新闻的语义层和知识层表示，并引入了预测点击率的注意机制

除了LibFM之外，所有的模型都是基于深度神经网络的。



![image-20200917164345777](https://i.loli.net/2020/09/17/b8lvkTMcnjzt4Jg.png)

（-）表示输入矩阵没有使用 profile embeddings，结果表明profile embedding可以提供完整的新闻信息进行推荐

基于协调过滤的DMF效果最差，由于新闻的时效性而不能很好地进行推荐。

### Discussion on different DAN variants

从以下几方面讨论DAN：使用实体和实体类型,转换函数的选择,注意机制的使用,选择集成ARNN组件的功能和使用

![image-20200917165249019](https://i.loli.net/2020/09/17/gcrzx1Rp2aF3h8j.png)

### Different dimension of entity embedding andentity-type embedding

作者的实验选参结果

![image-20200917165615379](https://i.loli.net/2020/09/17/FIpd3oeuiM4cv2L.png)

## 四、Related Work

1. collaborative filtering (CF) based methods：

   (Das et al. 2007; Marlin and Zemel 2004; Rendle 2010;Sheng,  Kawale,  and  Fu  2015;  Wu  et  al.  2016;  Xue  et  al.2017)

2. content based method：

    (IJntema et al. 2010; Kom-pan and Bielikov ́a 2010; Huang et al. 2013)

3. hybrid meth-ods ：

    (Morales,  Gionis,  and  Lucchese  2012;  Li  et  al.  2011;Liu,  Dolan,  and  Pedersen  2010)  

4. deep  learning  basedmethods：

    (Zheng, Noroozi, and Yu 2017; Wang et al. 2017)



疑问

1. 获取用户兴趣特征的时候，为什么需要考虑候选新闻，以及使用的ANN原理。

ANN利用注意力机制来捕捉用户点击新闻对候选新闻的不同影响，从而建立用户当前兴趣模型。

