# Investigating Capsule Networks with Dynamic Routing for Text Classification

文本建模方法大致可以分为两类：

（1）**忽略词序、对文本进行浅层语义建模**（代表模型包括 LDA，EarthMover’s distance等）; 

（2）**考虑词序、对文本进行深层语义建模**（深度学习算法，代表模型包括 LSTM，CNN 等）。

在神经网络方法中，在较低层次聚集的空间模式有助于表示较高层次的概念.

CNN 建立卷积特征检测器提取来自局部序列窗口的模式，并使用 max-pooling 来选择最明显的特征。然后，CNN 分层地提取不同层次的特征模式。

然而，**CNN 在对空间信息进行建模时，需要对特征检测器进行复制，降低了模型的效率**。

另一方面，**空间不敏感的方法在推理时间上是非常有效的,因为它不考虑任何词序或局部模式，但不可避免地受限于丰富的文本结构**（比如保存单词的位置信息、语义信息、语法结构等），**难以有效地进行编码且缺乏文本表达能力**。

胶囊网络（Capsule Network）, 用神经元向量代替传统神经网络的单个神经元节点，以 Dynamic Routing 的方式去训练这种全新的神经网络，有效地改善了上述两类方法的缺点。

正如在人类的视觉系统的推理过程中，可以智能地对局部和整体（part-whole）的关系进行建模，自动地将学到的知识推广到不同的新场景中

![截屏2020-11-23 下午9.44.17](/Users/mac/Library/Application Support/typora-user-images/截屏2020-11-23 下午9.44.17.png)

## N-gram Convolutional Layer

标准卷积层，提取一个句子序列中不同位置的n-gram特征

$x\in R^{L*V}$为输入句子的特征表示，L为句子长度，V是词向量维度

$x_i\in R^V $ 表示句子的第i个词       $W^a \in R^{K1*V}$为滤波器，K1为滑动窗口大小，

提取出的feature map为$m^a ∈ R^{L−K1 +1}$，其每个元素：
$$
m^a_i =f(x_{i:i+K1−1}◦W^a + b_0)
$$
◦为逐元素乘法，但应该得到标量，每次得到一个特征图应执行L−K1 +1次卷积操作，L−K1 +1次得到m



B个滤波器，得到：
$$
M = [m_1, m_2, ..., m_B] ∈ R^{(L−K1+1)×B}
$$
得到B个特征图，按模型理解这里应该每一行为一个特征图，下一层使用的$M_i$为一个特征图，作为一个胶囊，所有n-gram层得到一层胶囊列表

**没有直接在embedding上进行capsule的操作，原因可能是capsule对于高阶特征更有效**

## Primary Capsule Layer

胶囊用向量输出胶囊代替了CNNs的标量输出特征检测器，以保存实例化的参数，如单词的局部顺序和单词的语义表示.

$p_i \in R^{B*d} $为胶囊的参数，d为胶囊维数，对每个N-gram向量（$M_i \in R^B$）使用滤波器，$W_b \in R^{B*d}$为滤波器
$$
p_i=g(W^bM_i+b_1)
$$
此处应该是$(W^b)^TM_i$,生成相应的n-gram短语的胶囊形式，得到胶囊列表$p \in R^{(L-K1+1)*d}$

C个滤波器得到C个胶囊列表：（下面p对应一个列表而不是一个胶囊）
$$
P = [p_1, p_2, ..., p_C] ∈ R^{(L−K1+1)×C×d}
$$
得到(L − K1 + 1) × C 个d维向量（胶囊）

## generate prediction vector

1⃣️共享权重
$$
\hat{u}_{j|i} =W^{t1}_ju_i+ \hat{b}_{j|i} ∈Rd
$$
ui为底层胶囊，$W^{t1} \in R^{N*d*d}$共享权重，其中N为高层的胶囊数

2⃣️不共享权重

$W^{t2} ∈ R^{H ×N ×d×d} $,其中H为低层的胶囊数

![截屏2020-11-24 下午10.34.17](https://i.loli.net/2020/11/24/tDdlzEZgF4b9aGn.png)

## Dynamic Routing

提出**三种策略有减少背景或者噪音胶囊对网络的影响**：

1⃣️Orphan Category：捕捉文本的“背景”信息，如停止词和与特定类别无关的词，帮助胶囊网络更有效地建立模型

2⃣️Leaky-Softmax：替代Softmax

3⃣️Coefficients Amendment：加入children capsule的概率来计算耦合参数

![截屏2020-11-25 上午9.19.12](https://i.loli.net/2020/11/25/6sjHNFvohKIYn7p.png)

给定预测向量$\hat(u)_{j|i}$和child capsule的概率值$\hat{a}_{j|i} = a_i$,$c_{j|i}$更新为：
$$
c_{j|i} = \hat{a}_{j|i}.leakly-softmax(b_{j|i})
$$

## Convolutional Capsule Layer

在这一层中，每个胶囊仅在空间上连接到下面一层中的局部区域K2 * C

K2 · C为child capsule的数量，D为parent capsule的数量

假设$W^{c1} \in R^{D*d*d}$为共享权重，则预测向量计算为：
$$
\hat(u)_{j|i} = W^{c1}_j u_i+ b_{j|i}
$$
$u_i$是局部胶囊K2*C的child capsule，$W^{c1}_j$为第j个姿态矩阵，最后得到(L-K1-K2+2)xD个d维向量

## Fully Connected Capsule Layer

将Convolutional Capsule Layer的胶囊平整为胶囊列表，喂入全连接的胶囊层

转换矩阵为$W^{d1} \in R^{E*d*d}$ 或 $W^{d2} \in R ^{H*E*d*d}$

H是child capsule的数目，E为最后分类数+orphan 类别



## The Architectures of Capsule Network

![截屏2020-11-25 上午9.47.33](/Users/mac/Library/Application Support/typora-user-images/截屏2020-11-25 上午9.47.33.png)

Capsule-A从嵌入层开始，该嵌入层将语料库中的每个单词转换为一个300维（V = 300）的单词向量，然后是一个带有32个过滤器的3-gram（K1 = 3）卷积层（B = 32  ），并且步长为1（具有ReLU非线性）。 

其他层都是胶囊层，首先是具有32个过滤器的B×d初级胶囊层（C = 32），然后是具有16（D  = 16）个过滤器的3×C×d×d（K2 = 3）卷积胶囊层和全连接的胶囊层。
   每个胶囊都有16维（d = 16）的实例化参数，其长度（范数）可以描述胶囊存在的可能性。 胶囊层通过变换矩阵连接，并且每个连接还乘以路由系数，该路由系数是通过协议机制路由而动态计算的。

## Experiment

### dataset



![截屏2020-11-25 上午9.56.08](/Users/mac/Library/Application Support/typora-user-images/截屏2020-11-25 上午9.56.08.png)

### result

![截屏2020-11-25 上午9.59.33](https://i.loli.net/2020/11/25/7ezU6h1PmZ94sv3.png)

胶囊网络显著优于简单的深度神经网络，CNN、LSTM、Bi-LSTM

Capsule-B比Capsule-A要好，因为它可以学习更有意义和更全面的文本表示

## Ablation Study

![截屏2020-11-25 上午10.12.49](https://i.loli.net/2020/11/25/K18WlROFAoPGi2x.png)

仅仅针对作者修改的胶囊网络版本做的消融实验

w/o为without

## Single-Label to Multi-Label Text Classification

评估是在Reuters-21578数据集上进行的(Lewis, 1992)。该数据集由Reuters financial newswire service的10,788个文档组成，每个文档包含多个标签或单个标签。我们对语料库进行再处理，以评估胶囊网络从单标签到多标签的文本分类能力。

对于train和dev，我们只使用验证集和训练集中的单标签文档。

对于test，Reuters- Multi-label只使用测试数据集中的多标签文档，Reuters- full包含测试集中的所有文档

![截屏2020-11-25 上午11.03.30](https://i.loli.net/2020/11/25/VYtqCEzpNlkF7B6.png)



![截屏2020-11-25 上午11.03.01](https://i.loli.net/2020/11/25/6nJIHoOhTFm4KM5.png)

胶囊网络效果很好，因为胶囊网络能够保存由单标签文档训练的类别的实例化参数。与传统的深度神经网络相比，胶囊网络具有更强的传输能力。此外，Reuters-Full的良好结果也表明，胶囊网络在单标签文件方面比竞争对手具有强大的优势。

## Connection Strength Visualization

![截屏2020-11-25 上午10.42.36](/Users/mac/Desktop/截屏2020-11-25 上午10.42.36.png)

从Reuters-Multi-label test set选择了一个多标签文档，该文档的类别标签(如利率和货币/外汇)被我们的模型高度自信地正确预测(完全正确)(p > 0.8)

具体分类的短语，如“利率”和“外汇”用红色突出显示。

使用标签云来可视化用于利率和货币/外汇类别的3-gram短语，连接强度越强，字体越大。从分析结果可以看出，胶囊网络能够根据文本类别正确地对重要短语进行编码和聚类。用直方图表示主胶囊与全连通胶囊之间的连接强度，如表6(底线)所示。由于空间的原因，五幅柱状图被分割。路由程序正确地将选票路由到利率和货币/外汇类别。