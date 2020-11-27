# MATRIX CAPSULES WITH EM ROUTING

每个胶囊都有一个逻辑单元来表示实体的存在和一个4x4矩阵可以学习表示实体和观察者之间的关系(姿态)

![截屏2020-11-08 下午9.14.36](https://i.loli.net/2020/11/08/5qMou8z4OgFiUZ7.png)

ReLU Conv1：标准的卷积层，kernel = 5*5, channels=A=32。

PrimaryCaps：

