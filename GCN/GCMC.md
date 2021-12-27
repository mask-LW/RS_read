# **Graph Convolutional Matrix Completion**



推荐系统的一个子任务就是矩阵补全。文中把矩阵补全视作在图上的链路预测问题：users和items的交互数据可以通过一个在user和item节点之间的二分图来表示，每种链接边可以看做是一种label（例如多种交互行为：点击，收藏，喜欢，下载等；在评分中，1~5分可以分别看做一种label）。其中观测到的评分/购买用links来表示。因此，预测评分就相当于预测在这个user-item二分图中的links。

作者提出了一个图卷积矩阵补全（GCMC）框架：在用深度学习处理图结构的数据的研究进展的基础上，对矩阵进行补全的一种图自编码器框架。**这个自编码器通过在二部交互图中信息传递的形式生成user和item之间的隐含特征**。这种user和item之间的隐含表示用于通过一个双线性的解码器重建评分links。

**当推荐图中带有结构化的外部信息(如社交网络)时，将矩阵补全作为二部图上的链接预测任务的好处就变得尤为明显。将这些外部信息与交互数据相结合可以缓解与冷启动问题相关的性能瓶颈**。







## **Matrix completion as link prediction in bipartite graphs**

![截屏2021-06-06 下午5.56.43](https://i.loli.net/2021/06/06/q4yn1hYljt6uMzi.png)



评分矩阵M：$N_u*N_v$，分别为用户和item的数量

非0的$M_{ij}$表示user i对 item j的评分，0表示没有观测到评分

user和item的交互用无向图G表示：![截屏2021-06-06 下午6.00.54](https://i.loli.net/2021/06/06/ysdQpIhT5ZJOHrF.png)

其中$W_u$和$W_v$分别表示user和item的集合



## **Graph auto-encoders**

编码器：Z = f(X,A)

输入：**N** ×**D** 的特征矩阵和邻接矩阵A

输出：N x E的结点矩阵 $Z = 【z^T_1,Z^t_2,...Z^T_N】$



解码器： $\hat{A} = g(Z)$

输入： 结点对 $(z_i,z_j)$

输出：预测邻接矩阵的实体 $\hat{A}_{ij}$

N表示结点数，D表示输入特征数，E为嵌入维度



将编码器公式化：二部图G可以看作![截屏2021-06-07 上午10.41.13](https://i.loli.net/2021/06/07/ZYgvOj4XAhIUGy3.png)

其中前两个元素为user和item的嵌入矩阵，分别为 $N_u * E , N_v*E$

$M_r \in {0,1}^{N_u*N_v}$是和交互类型r相关的邻接矩阵



将解码器公式化：$\hat{M} = g(U,V)$表示作用在user和item的嵌入的函数，返回一个评分矩阵M，维度为$N_u*N_v$



通过最小化$\hat{M}$的预测评分和M中观测到真实评分的重构误差来进行训练，误差可以使用均方根误差或交叉熵损失



## **Graph convolutional encoder**



作者首先提出了一种针对不同rating level的sub-graph分别进行局部图卷积，再进行汇聚的encoder模型。这种局部图卷积可以看做是一种信息传递(message passing)，即：向量化形式的message在图中不同的边上进行传递和转换。

以item j 到user i 传递信息：

![截屏2021-06-07 上午10.53.13](https://i.loli.net/2021/06/07/EcwQAWgOvN5CHFn.png)

针对每种类型的边，对应user i的邻居节点进行累加操作：

![截屏2021-06-07 上午10.55.56](https://i.loli.net/2021/06/07/XAZKUDpCOT3jr9m.png)

将不同类型的边采集到的信息作汇聚：
![截屏2021-06-07 上午10.59.09](https://i.loli.net/2021/06/07/FcU67fyaZLVmYoC.png)

accum为汇聚操作，可为stack或sum，可选择ReLU函数

为了得到用户最终的嵌入，作者加了个全连接层做了个变换：

![截屏2021-06-07 上午11.01.17](https://i.loli.net/2021/06/07/l6iIRMAfv9Fb4on.png)

公式（2）为卷积层，（3）为dense层，可适当叠加（2），但最初的实验发现（2）结合（3）的性能最佳

##  **Bilinear decoder**



为了重构二分图上的链接，作者采用了bilinear decoder，把不同的rating level分别看做一个类别。bilinear operation后跟一个softmax输出每种类别的概率：

![截屏2021-06-07 上午11.05.44](https://i.loli.net/2021/06/07/YKhsktlxMa4GZAR.png)

Q是可训练的ExE参数矩阵，E是user或item隐含特征的维度

则最终预测的评分是关于上述概率分布的期望：

![截屏2021-06-07 上午11.06.38](https://i.loli.net/2021/06/07/TcCfJkY4ZXy1rHA.png)



##  **Model training**

### **Loss function** 

最小化预测评分的负数对数似然：

![截屏2021-06-07 上午11.08.40](https://i.loli.net/2021/06/07/PzxNBCrkT9hVq4R.png)



$I[k=l]=1$表示当k=l时值为1否则为0

此处的![截屏2021-06-07 上午11.11.53](https://i.loli.net/2021/06/07/1BV9cjfSF42eoJ6.png)

作为未观察到的评分等级的mask，这样对于在M中的元素，如果值为1就代表观测到了的评分等级，为0则对应未观测到的。因此，只需要优化观测到的评分等级。

### Node dropout

为了使该模型能够很好地泛化到未观测到的评分等级，在训练中使用了dropout 以一定概率随机删除特定节点的所有传出消息，将其称为节点dropout。

### Mini-batching

只采样固定数量的user和item对

对于除了MovieLens-10M之外的所有数据集，选择full batches训练，因为mini-batches训练收敛更快

###  **Weight sharing**

不是所有的用户或物品在不同rating level上都拥有相等的评分数量。这会导致在局部卷积时，Wr上某些列可能会比其他列更少地被优化。因此需要使用某种在不同的r之间进行参数共享的方法：

![截屏2021-06-07 上午11.51.52](https://i.loli.net/2021/06/07/rLRAVCP7aiWlIwp.png)

Ts是基础矩阵。也就是说越高评分，Wr包含的Ts数量越多。

作者采用了一种基于基础参数矩阵的线性组合的参数共享方法：

![截屏2021-06-07 上午11.52.46](https://i.loli.net/2021/06/07/joh375aLqbiCk8A.png)

###   **Input feature representation and side information**

当内容表征不足与区分用户或item，为了建模结点的辅助信息，作者在对h做全连接层变换时，考虑了结点的辅助信息:

![截屏2021-06-07 上午11.47.59](https://i.loli.net/2021/06/07/fQOVIxBXELhtz28.png)

即：先对特征x作一个变换得到f，与h分别经过线性变换并激活得到最终对表示

## **Experiments**

数据集

![截屏2021-06-07 上午11.16.15](https://i.loli.net/2021/06/07/mT3UC7NKj9IOW4x.png)



​		

RMSE均方根误差：

![截屏2021-06-07 上午11.55.48](https://i.loli.net/2021/06/07/h4D8IvG3lKejqRs.png)

![截屏2021-06-07 上午11.17.25](https://i.loli.net/2021/06/07/vclZME8VAqoOsyn.png)

在二部交互图上进行信息传递的简单的自编码模型比更复杂的递归估计有更好的性能



![截屏2021-06-07 上午11.17.50](https://i.loli.net/2021/06/07/S4U5XvWarPuQRHk.png)



![截屏2021-06-07 下午12.05.04](https://i.loli.net/2021/06/07/LvMI2KjSlepr4WN.png)

## 总结

### 核心流程

![截屏2021-06-07 下午12.07.34](https://i.loli.net/2021/06/07/5MSLDvbJfWGTYF1.png)











