# Self-supervised Graph Learning for Recommendation

核心的思想是，在传统监督任务的基础上，增加辅助的**自监督学习任务**，变成多任务学习的方式。

具体而言，同一个结点先通过**数据增强** (data augmentation)的方式产生多种视图(multiple views)，然后借鉴**对比学习**(contrastive learning)的思路，最大化同一个结点不同视图表征之间的相似性，最小化不同结点表征之间的相似性，实际上是一个**self-discrimination**的任务。对于数据增强，对同一个结点，为了产生不同的视图，使用了3种方式，从不同维度来改变图的结构。包括，结点维度的node dropout；边维度的edge dropout（能够降低高度结点的影响力）；图维度的random walk。这种图结构的扰动和数据增强操作，能够提高模型对于噪声交互的鲁棒性。





![截屏2021-06-05 下午5.23.28](https://i.loli.net/2021/06/05/xm2EGqv5TDUugow.png)