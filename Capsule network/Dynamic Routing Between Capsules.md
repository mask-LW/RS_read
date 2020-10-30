# Dynamic Routing Between Capsules

Authors: Sabour, Sara、 Nicholas Frosst、Hinton, Geoffrey E

**publication**：31st Conference on Neural Information Processing Systems (NIPS 2017), Long Beach, CA, USA.

 Cited times: 1947

在图像识别领域，目前是各种CNN网络的天下，不过CNN网络有其自身的局限性，这一点Hinton之前也提到过。CNN的不足主要体现在下面两方面 ：

1. CNN中，对不同特征的相对位置并不关注。
2. 池化操作虽然效果很好，增加了模型的鲁棒性，但同时丢失了模型的很多细节。 

所谓“胶囊（capsules）”指的是人脑中的一种结构，它们能够很好的处理不同类型的视觉刺激并对诸如位置、形制、速度等信息进行编码。

## How the vector inputs and outputs of a capsule are computed

Neuron：output a value

Capsule: output a vector

需要一个胶囊的输出向量的长度来表示这个胶囊所表示的实体出现在当前输入中的概率，使用一个非线性“压缩”函数来确保短向量缩小到几乎为零，长向量缩小到略低于1的长度：
$$
v_j = \frac{||s_j||^2}{1+||s_j||^2}\frac{s_j}{||s_j||},(1)
$$
其中，$$v_j$$是capsule j的向量输出，$s_j$是它的总输入

除第一层以外，capsule $s_j$ 的输入由所有“prediction vectors”  $\hat {u}_{j|i}$  进行加权和得到，而$\hat {u}_{j|i}$  由权重矩阵$W_{ij}$和capsule 的输出$u_i$相乘得到：
$$
S_j = \sum_i cij \hat {u}_{j|i} \qquad,\qquad \hat {u}_{j|i} = W_{ij}u_i,(2)
$$

$\hat {u}_{j|i}$ 表示上一层第 i 个 Capsule 的输出向量和对应的权重向量相乘（$W_{ij}$ 表示向量而不是元素）而得出的预测向量,也可以理解为在前一层为第 i 个 Capsule 的情况下连接到后一层第 j 个 Capsule 的强度.

$c_{ij}$ 是 iterative dynamic routing process得到的耦合系数，耦合系数之和为1，由一个“routing softmaax”确定，其初始logits $b_{ij}$是胶囊i与胶囊j耦合的log先验概率
$$
c_{ij} = \frac{exp(b_{i_j})}{\sum_kexp(b_{ik})},(3)
$$
![截屏2020-10-25 下午8.36.36](https://i.loli.net/2020/10/25/eBqfwKLbFVJnTAM.png)



## Margin loss for digit existence

使用实例化向量的长度来表示胶囊实体存在的概率,当且仅当映像中的数字出现时，我们希望数字分类的顶级胶囊有一个很长的实例化向量，对每个数字胶囊k使用一个单独的margin loss $L_k$:
$$
L_k = T_k max(0,m^+-||v_k||)^2 + \lambda(1-T_k) max(0,||v_k||-m^-)^2
$$
如果有一个k类的数字，m+= 0.9, m−= 0.1,$T_k$= 1.



## CapsNet architecture

简单的capsne结构：两个卷积层和一个完全连接层



![截屏2020-10-25 下午10.02.25](https://i.loli.net/2020/10/25/InirQFOvdwVUtfP.png)

输入图片的维度为28x28

1⃣️第一个卷积层Conv1具有256,9×9的卷积核，步长为1,ReLU激活。这一层将像素强度转换为局部特征检测器的活动，然后将其用作主胶囊的输入。经过Conv得到（28-9+1）x (28-9+1) x256的张量，此处应该没有使用padding，需要参数9x9x256+256= 20992

主胶囊是多维实体的最低层，从反图形的角度来看，激活主胶囊相当于反转渲染过程。与将实例化的部分拼凑成熟悉的整体相比，这是一种非常不同的计算类型，而这正是胶囊所擅长的。

2⃣️第二个卷积层开始作为 Capsule 层的输入而构建相应的张量结构

第二层(PrimaryCapsules)是一种卷积胶囊层32通道的卷积8 d胶囊(即每个主要胶囊包含8卷积单位9×9的内核和2的stride）。

考虑 32 个（32 channel）9×9 的卷积核在步幅为 2 的情况下做卷积，那么实际上得到的是传统的 6×6×32 的张量，即等价于 6×6×1×32，而 PrimaryCaps 的输出需要是一个长度为 8 的向量，因此传统卷积下的三维输出张量 6×6×1×32 就需要变化为四维输出张量 6×6×8×32。

因此将第二个卷积层看作对维度为 20×20×256 的输入张量执行 8 次不同权重的 Conv2d 操作，每次 Conv2d 都执行带 32 个 9×9 卷积核、步幅为 2 的卷积操作。

![截屏2020-10-26 下午9.23.22](https://i.loli.net/2020/10/26/DtylLfVzkHSgdO5.png)

每次卷积操作都会产生一个 6×6×1×32 的张量，一共会产生 8 个类似的张量，那么将这 8 个张量（即 Capsule 输入向量的 8 个分量）在第三个维度上合并在一起就成了 6×6×8×32。从上可知 PrimaryCaps 就相当于一个深度为 32 的普通卷积层，只不过每一层由以前的标量值变成了长度为 8 的向量。

PrimaryCaps 层的卷积计算都没有使用 ReLU 等激活函数，它们以向量的方式预备输入到下一层 Capsule 单元中。

PrimaryCaps 每一个向量的分量层级是共享卷积权重的，即获取 6×6 张量的卷积核权重为相同的 9×9 个。这样该卷积层的参数数量为 9×9×256×8×32+8×32=5308672，其中第二部分 8×32 为偏置项参数数量。

3⃣️最后一层(DigitCaps)：第三层 DigitCaps 在第二层输出的向量基础上进行传播与 Routing 更新。第二层共输出 6×6×32=1152 个向量，每一个向量的维度为 8，即第 i 层共有 1152 个 Capsule 单元。而第三层 j 有 10 个标准的 Capsule 单元，每个 Capsule 的输出向量有 16 个元素。前一层的 Capsule 单元数是 1152 个，那么 $w_ij$ 将有 1152×10 个，且每一个 $w_ij $ 的维度为 8×16。

当  $u_i$ 与对应的  $w_{ij}$  相乘得到预测向量后，我们会有 1152×10 个耦合系数 c_ij，对应加权求和后会得到 10 个 16×1 的输入向量。将该输入向量输入到「squashing」非线性函数中求得最终的输出向量 $v_j$，其中 v_j 的长度就表示识别为某个类别的概率。

DigitCaps 层与 PrimaryCaps 层之间的参数包含两类，即 W_ij 和 c_ij。所有 W_ij 的参数数量应该是 6×6×32×10×8×16=1474560，c_ij 的参数数量为 6×6×32×10×16=184320，此外还应该有 2×1152×10=23040 个偏置项参数,共有参数： 5537024

只在两个连续的胶囊层(如PrimaryCapsules和DigitCaps)之间进行路由。因为Conv1的输出是1d，所以在它的空间中没有需要一致的方向。因此，在Conv1和PrimaryCapsules之间不使用路由。

### 损失函数与优化

耦合系数 $c_{ij}$是通过一致性 Routing 进行更新的，他并不需要根据损失函数更新，但整个网络其它的卷积参数和 Capsule 内的 $W_{ij}$ 都需要根据损失函数进行更新。一般我们就可以对损失函数直接使用标准的反向传播更新这些参数，作者采用了 SVM 中常用的 Margin loss
$$
L_k = T_k max(0,m^+-||v_k||)^2 + \lambda(1-T_k) max(0,||v_k||-m^-)^2
$$
其中 k 是分类类别，$$T_k$$ 为分类的指示函数（k 存在为 1，k 不存在为 0），m+ 为上边界，m- 为下边界。此外，$v_k$的模即向量的 L2 距离。

实例化向量的长度来表示 Capsule 要表征的实体是否存在，所以当且仅当图片里出现属于类别 k 的手写数字时，我们希望类别 k 的最顶层 Capsule 的输出向量长度很大（在本论文 CapsNet 中为 DigitCaps 层的输出）。为了允许一张图里有多个数字，我们对每一个表征数字 k 的 Capsule 分别给出单独的 Margin loss。

构建完损失函数，我们就能愉快地使用反向传播了。

### Reconstruction as a regularization method

![截屏2020-10-25 下午10.12.06](https://i.loli.net/2020/10/25/TeKmzVp76LGjZrh.png)

使用一个额外的重构损失来鼓励数字胶囊对输入数字的实例化参数进行编码。

在训练期间，除了正确的数字胶囊的活动向量外，我们将其全部隐藏起来。

![截屏2020-10-26 上午9.35.35](https://i.loli.net/2020/10/26/Uds6pRVyMWKIrkP.png)

CapsNet的16D输出重构是稳健的，但只保留了重要的细节



## Capsules on MNIST

![截屏2020-10-25 下午10.19.16](https://i.loli.net/2020/10/25/uzdgImNMKc1kbpQ.png)

实验模型都是在在28×28MNIST的图像上进行训练，这些图像在每个方向上被移动了2个像素，并且有零填充。

基线是一个标准的CNN，三个卷积层，256,256,128个channels。每个有5x5kernels和1stride。最后的卷积层之后是两个大小为328,192的完全连接层。最后一个全连接层用dropout连接到一个有交叉熵损失的10级softmax层。

### What the individual dimensions of a capsule represent

由于我们只传递一个数字的编码，并对其他数字进行零化，因此数字胶囊的尺寸应该学会在实例化该类的数字时跨越变化的空间。这些变化包括笔画厚度、斜度和宽度。它们还包括特定于digit的变化，比如2的尾巴的长度。通过使用解码器网络，我们可以看到各个维度代表什么。

可以将这个活动向量的扰动版本输入译码器网络，看看扰动是如何影响重构的

![截屏2020-10-26 上午9.51.33](https://i.loli.net/2020/10/26/UwBz3cQlsoiajrb.png)

向量的某个维度代表了该向量的具体信息，通过改动查看某个维度对应哪个信息，比如笔画、旋转。

### Robustness to Affine Transformations

实验表明，与传统卷积网络相比，每个DigitCaps胶囊对每个类学习了一个更健壮的表示。由于手写数字在倾斜、旋转、样式等方面存在自然方差，因此经过训练的CapsNet对训练数据的小仿射变换具有中等鲁棒性

## Segmenting highly overlapping digits

3层CapsNet模型是在MultiMNIST训练数据上从零开始训练的，比基线卷积模型获得了更高的测试分类准确率

![截屏2020-10-26 上午10.24.05](https://i.loli.net/2020/10/26/JTSMqOYaIzfipwk.png)

显示胶囊网络在分割重叠数字方面提供了很好的性能





## why Capsule work

![截屏2020-10-26 上午10.45.20](https://i.loli.net/2020/10/26/UcIAOYqQVo5G8N3.png)

经过处理但代表同一数字的图片通过传统的神经网络输出得到的是同样的向量。

经过处理但代表同一数字的图片通过Capsule输出得到的是不同的向量，而且这两个向量之间的关系反映了这两张图片经过处理的关系，比如翻转。

当图像进行一些旋转、变形或朝向不同的方向，那么 CNN 本身是无法处理这些图片的，但可以在训练中添加相同图像的不同变形而得到解决。

![截屏2020-10-26 上午10.49.15](https://i.loli.net/2020/10/26/Z3dwCGIXKkv2oQB.png)

CNN只能做到Invariance，即最后的输出无法知道同样含义的输入之间的差距。

Capsule最后输出的向量的范数都是相同的，但向量内部的维度能够反映输入的差异，而最后的结果选择无视这个差异来预测结果，从而实现了equivariance。



Props：

1⃣️在MINST实现state of art

2⃣️相比传统CNN要求更少的训练数据

3⃣️可以检测出图片的位置和姿态信息

4⃣️在数字重叠方面有很好的性能

5⃣️为仿射变换提供很好的鲁棒性

Cons：

1⃣️在CIFAR10没有达到state of art

2⃣️没有在imagenet等更大的图片集上进行测试

3⃣️Dynamic routing里面嵌套多重循环，训练时间较长





问题：

其中运算的参数c只通过Dynamic routing来获得，没有做对比实验确定通过back propogation得到参数c而实现的效果不佳。





李宏毅对Capsule的解说：



![截屏2020-10-25 下午8.26.34](https://i.loli.net/2020/10/26/V8SREfxmCDXh1Yi.png)



![截屏2020-10-25 下午8.35.49](/Users/mac/Library/Application Support/typora-user-images/截屏2020-10-25 下午8.35.49.png)



![截屏2020-10-25 下午9.49.05](https://i.loli.net/2020/10/26/6C2SbKLcAm7zrgq.png)

![截屏2020-10-25 下午9.54.57](https://i.loli.net/2020/10/26/tbZ1Nis5wmSWu4D.png)

1⃣️capsule可以简单地替代filter

2⃣️直接作为output layer输出，以10数字分类为例，原来的神经网络输出一个向量，比如[1,0,0,0,0,0,0,0,0,0],但使用capsule输出时需要10个capsule，分别输出对应0～9的向量，原来论文还加入NN进行reconstruction，即预测的数字对应向量x1，其他向量x0





squash函数：

![截屏2020-10-26 下午3.52.31](https://i.loli.net/2020/10/26/aT5iRJAGnjIMvr2.png)



画图表示为：

![截屏2020-10-26 下午3.53.18](https://i.loli.net/2020/10/26/5w6fK23mnzSvYCg.png)

1. 值域在[0,1]之间，所以输出向量的长度可以表征某种概率。
2. 函数单调增，所以“鼓励”原来较长的向量，而“压缩”原来较小的向量，即长向量趋于1，短向量趋于0

也就是 Capsule 的“激活函数” 实际上是对向量长度的一种压缩和重新分布。



实现：

pytorch版本：https://github.com/gram-ai/capsule-networks

tensorflow版本：https://github.com/naturomics/CapsNet-Tensorflow

keras版本：https://github.com/XifengGuo/CapsNet-Keras





geron：





![截屏2020-10-26 下午4.16.47](https://i.loli.net/2020/10/26/bILrwWxzeFUy3ME.png)

在计算机视觉中，用一些参数渲染一个对象，比如表示一个图形，使用三角形的位置坐标和角度等参数。



![截屏2020-10-26 下午4.15.59](/Users/mac/Library/Application Support/typora-user-images/截屏2020-10-26 下午4.15.59.png)

逆图形，从图片入手，找出图像包含的对象及其包含的实例化参数。

一个胶囊网络基本上是一个试图执行反向图形解析的神经网络。



![截屏2020-10-26 下午4.22.47](https://i.loli.net/2020/10/26/6GmXVzBtLkAKlYT.png)



例如，上面的网络包含50个胶囊。 箭头表示这些胶囊的输出向量。 胶囊输出许多向量。 黑色箭头对应于试图找到矩形的胶囊，而蓝色箭头则表示胶囊寻找三角形的输出。激活向量的长度表示胶囊正在查找的物体确实存在的估计概率。

你可以看到大多数箭头很小，这意味着胶囊没有检测到任何东西，但是两个箭头相当长。 这意味着在这些位置的胶囊非常有自信能够找到他们要寻找的东西，在这个情况下是矩形和三角形。

此时一个输出向量的范数代表该对象存在的概率，然后编码对象的实例化参数，例如在这个情况下，对象的旋转，但也可能是它的厚度，它是如何拉伸或倾斜的，它的确切位置（可能有轻微的翻转），等等





![截屏2020-10-26 下午4.27.20](/Users/mac/Library/Application Support/typora-user-images/截屏2020-10-26 下午4.27.20.png)

实现这一点的一个好方法是首先应用一对卷积层，就像在常规的卷积神经网络中一样。这将输出一个包含一堆特征映射的数组。 然后你可以重塑这个数组来获得每个位置的一组向量。

例如，假设卷积图层输出一个包含18个特征图（2×9）的数组，则可以轻松地重新组合这个数组以获得每个位置9个维度的2个向量。 你也可以得到3个6维的向量，等等。

然后应用一个squashing（压扁）函数。它保留了矢量的方向，但将它压扁，以确保它的长度在0到1之间。

🌟胶囊网络的一个关键特性是在网络中保存关于物体位置和姿态的详细信息。例如，如果我稍微旋转一下图像，注意激活向量也会稍微改变，叫做equivariance。



参考：

【1】https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650732855&idx=1&sn=87319e9390200f24dfd2faff4d7d364a&chksm=871b3d49b06cb45fd8a68d003310b05562d9f8ff094ed08345f112e4450f7e66e6cf71c5b571&scene=21#wechat_redirect

【2】https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650731207&idx=1&sn=db9b376df658d096f3d1ee71179d9c8a&chksm=871b36b9b06cbfafb152abaa587f6730716c5069e8d9be4ee9def055bdef089d98424d7fb51b&scene=21#wechat_redirect

```python
CapsuleNet(
  (conv1): Conv2d(1, 256, kernel_size=(9, 9), stride=(1, 1))
  (primary_capsules): CapsuleLayer(
    (capsules): ModuleList(
      (0): Conv2d(256, 32, kernel_size=(9, 9), stride=(2, 2))
      (1): Conv2d(256, 32, kernel_size=(9, 9), stride=(2, 2))
      (2): Conv2d(256, 32, kernel_size=(9, 9), stride=(2, 2))
      (3): Conv2d(256, 32, kernel_size=(9, 9), stride=(2, 2))
      (4): Conv2d(256, 32, kernel_size=(9, 9), stride=(2, 2))
      (5): Conv2d(256, 32, kernel_size=(9, 9), stride=(2, 2))
      (6): Conv2d(256, 32, kernel_size=(9, 9), stride=(2, 2))
      (7): Conv2d(256, 32, kernel_size=(9, 9), stride=(2, 2))
    )
  )
  (digit_capsules): CapsuleLayer()
  (decoder): Sequential(
    (0): Linear(in_features=160, out_features=512, bias=True)
    (1): ReLU(inplace=True)
    (2): Linear(in_features=512, out_features=1024, bias=True)
    (3): ReLU(inplace=True)
    (4): Linear(in_features=1024, out_features=784, bias=True)
    (5): Sigmoid()
  )
)

```











