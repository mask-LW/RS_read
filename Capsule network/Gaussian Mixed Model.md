# Gaussian Mixed Model（高斯混合模型）

## **混合模型（Mixture Model）**

混合模型是一个可以用来表示在总体分布（distribution）中含有 K 个子分布的概率模型，换句话说，混合模型表示了观测数据在总体中的概率分布，它是一个由 K 个子分布组成的混合分布。混合模型不要求观测数据提供关于子分布的信息，来计算观测数据在总体分布中的概率。

## **高斯模型**

### **单高斯模型**

正态分布 = 高斯分布 = 常态分布

当样本数据 X 是一维数据（Univariate）时，高斯分布遵从下方概率密度函数（Probability Density Function）：【正态分布的概率密度函数】

![[公式]](https://www.zhihu.com/equation?tex=P(x|\theta)+%3D+\frac{1}{\sqrt{2\pi\sigma^{2}}}+exp(-\frac{(x-\mu)^2}{2\sigma^{2}}))

其中 ![[公式]](https://www.zhihu.com/equation?tex=\mu) 为数据均值（期望）， ![[公式]](https://www.zhihu.com/equation?tex=\sigma) 为数据标准差（Standard deviation）。

当样本数据 X 是多维数据（Multivariate）时，高斯分布遵从下方概率密度函数：

![[公式]](https://www.zhihu.com/equation?tex=P(x|\theta)+%3D+\frac{1}{(2\pi)^{\frac{D}{2}}\left|+\Sigma+\right|^{\frac{1}{2}}}exp(-\frac{(x-\mu)^{T}\Sigma^{-1}(x-\mu)}{2}))

其中， ![[公式]](https://www.zhihu.com/equation?tex=\mu) 为数据均值（期望）， ![[公式]](https://www.zhihu.com/equation?tex=\Sigma) 为协方差（Covariance），D 为数据维度。

### **高斯混合模型**

高斯混合模型可以看作是由 K 个单高斯模型组合而成的模型，这 K 个子模型是混合模型的隐变量（Hidden variable）。一般来说，一个混合模型可以使用任何概率分布，这里使用高斯混合模型是因为高斯分布具备很好的数学性质以及良好的计算性能。

有一组狗的样本数据，不同种类的狗，体型、颜色、长相各不相同，但都属于狗这个种类，此时单高斯模型可能不能很好的来描述这个分布，因为样本数据分布并不是一个单一的椭圆，所以用混合高斯分布可以更好的描述这个问题，如下图所示：

![img](https://pic1.zhimg.com/80/v2-b1a0d985d1508814f45234bc98bf9120_720w.jpg)

图中每个点都由 K 个子模型中的某一个生成

首先定义如下信息：

- ![[公式]](https://www.zhihu.com/equation?tex=x_{j}) 表示第 ![[公式]](https://www.zhihu.com/equation?tex=j) 个观测数据， ![[公式]](https://www.zhihu.com/equation?tex=j+%3D+1%2C2%2C...%2CN)
- ![[公式]](https://www.zhihu.com/equation?tex=K) 是混合模型中子高斯模型的数量， ![[公式]](https://www.zhihu.com/equation?tex=k+%3D+1%2C2%2C...%2CK)
- ![[公式]](https://www.zhihu.com/equation?tex=\alpha_{k}) 是观测数据属于第 ![[公式]](https://www.zhihu.com/equation?tex=k) 个子模型的概率， ![[公式]](https://www.zhihu.com/equation?tex=\alpha_{k}+\geq+0) ， ![[公式]](https://www.zhihu.com/equation?tex=\sum_{k%3D1}^{K}{\alpha_{k}}+%3D+1)
- ![[公式]](https://www.zhihu.com/equation?tex=\phi(x|\theta_{k})) 是第 ![[公式]](https://www.zhihu.com/equation?tex=k) 个子模型的高斯分布密度函数， ![[公式]](https://www.zhihu.com/equation?tex=\theta_{k}+%3D+(\mu_{k}%2C+\sigma_{k}^{2})) 。其展开形式与上面介绍的单高斯模型相同
- ![[公式]](https://www.zhihu.com/equation?tex=\gamma_{jk}) 表示第 ![[公式]](https://www.zhihu.com/equation?tex=j) 个观测数据属于第 ![[公式]](https://www.zhihu.com/equation?tex=k) 个子模型的概率

高斯混合模型的概率分布为：

![[公式]](https://www.zhihu.com/equation?tex=P(x|\theta)+%3D+\sum_{k%3D1}^{K}{\alpha_{k}\phi(x|\theta_{k})})

对于这个模型而言，参数 ![[公式]](https://www.zhihu.com/equation?tex=\theta+%3D+(\tilde{\mu_{k}}%2C+\tilde{\sigma_{k}}%2C+\tilde{\alpha_{k}})) ，也就是每个子模型的期望、方差（或协方差）、在混合模型中发生的概率。













具体来说，对于已有的向量$$x_1,…,x_n$$，GMM希望找到它们所满足的分布$$p(x)$$

GMM设想这批数据能分为几部分（类别），每部分单独研究，也就是:
$$
p(x) = \sum ^k_{j=1} p(j)p(x|j)
$$
其中j代表了类别，取值为1,2,…,k，由于p(j)p(j)跟**x**x没关系，因此可以认为它是个常数分布，记$$p(j)=πj$$。

p(x|j)就是这个类内的概率分布,x是一个向量，取最简单的高斯分布
$$
N(x;μ_j,Σ_j)=\frac{1}{(2π)^{d/2}(detΣ_j)^{1/2}} exp(\frac{−1}{2}(x−μj)⊤Σ^{−1}_j(x−μj))
$$
**GMM的特性就是用概率分布来描述一个类**

其中d是向量x的分量个数。现在我们得到模型的基本形式
$$
p(x)=∑_{j=1}^kp(j)×p(x|j)=∑_{j=1}^k π_j N(x;μ_j,Σ_j)
$$
