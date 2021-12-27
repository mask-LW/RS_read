# Pytorch_note

## Torch.nn

### 1.`torch.nn.parameter()`

将一个固定不可训练的tensor转换成可以训练的类型parameter，并将这个parameter绑定到这个module里面(net.parameter()中就有这个绑定的parameter，所以在参数优化的时候可以进行优化的)，所以经过类型转换这个self.v变成了模型的一部分，成为了模型中根据训练可以改动的参数了。使用这个函数的目的也是想让某些变量在学习的过程中不断的修改其值以达到最优化。

```python
self.v = torch.nn.Parameter(torch.FloatTensor(hidden_size))

```



### 2.`torch.nn.Linear()`

对输入数据做线性变换：$y=xA^T+b$

linear里面的weight和bias就是parameter类型，且不能够使用tensor类型替换，还有linear里面的weight甚至可能通过指定一个不同于初始化时候的形状进行模型的更改。一般是多维的可训练tensor。

参数：

**in_features** – size of each input sample

**out_features** – size of each output sample

**bias** – If set to `False`, the layer will not learn an additive bias. Default: `True`

shape：
$$
Input: (N, *, H_{in}) *代表其它维度，而且 H_{in}为输入的维度
$$

$$
Ouput: (N, *, H_{out}) *代表其它维度，而且 H_{out}为输出的维度
$$

Examples:

```python
>>> m = nn.Linear(20, 30)
>>> input = torch.randn(128, 20)
>>> output = m(input)
>>> print(output.size())
torch.Size([128, 30])

```

### 3.`torch.nn.Conv1d()`

class torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)

- in_channels(`int`) – 输入信号的通道。在文本分类中，即为词向量的维度
- out_channels(`int`) – 卷积产生的通道。有多少个out_channels，就需要多少个1维卷积
- kernel_size(`int` or `tuple`) - 卷积核的尺寸，卷积核的大小为(k,)，第二个维度是由in_channels来决定的，所以实际上卷积大小为kernel_size*in_channels
- stride(`int` or `tuple`, `optional`) - 卷积步长
- padding (`int` or `tuple`, `optional`)- 输入的每一条边补充0的层数
- dilation(`int` or `tuple`, `optional``) – 卷积核元素之间的间距
- groups(`int`, `optional`) – 从输入通道到输出通道的阻塞连接数
- bias(`bool`, `optional`) - 如果`bias=True`，添加偏置

Input size:（N，Cin，L）

Output size:（N，Cout，L）
$$
out(N_i,C_{out_j}) = bias(C_{out_j}) + \sum^{C_{in}-1}_{k=0}weight(C_{out_j},k)*input(N_i,k)
$$
N是batch size，C是channel维度，L是序列长度

Examples:

```python
m = nn.Conv1d(16, 33, 3, stride=2)
input = torch.randn(20, 16, 50)
output = m(input)
print(output.size())


torch.Size([20, 33, 24])
```

可以理解为中间为输入的维度变化，第三个为输入长度变化

### 4.`torch.nn.ReLU()`

按元素应用非线性函数，输入和输出的维度一致

### 5.`torch.bmm(input, mat2, *, deterministic=False, out=None) → Tensor`

实现batch的矩阵乘法,第一个维度b为batch size，两个tensor的维度必须为3.

If `input` is a (*b*×*m*×*p*) tensor, `out` will be a (*b*×*n*×*p*) tensor.

Examples:

```python
>>> input = torch.randn(10, 3, 4)
>>> mat2 = torch.randn(10, 4, 5)
>>> res = torch.bmm(input, mat2)
>>> res.size()
torch.Size([10, 3, 5])
```

### 6.`torch.nn.Dropout(*p: float = 0.5*, *inplace: bool = False*)`

输入和输出的维度一致，防止过拟合

Examples:

```python
m = nn.Dropout(p=0.2)
input = torch.randn(20, 16)
output = m(input)
print(output.size())

torch.Size([20, 16])
```



### 7.`torch.nn.NLLLoss

CrossEntropyLoss()=log_softmax() + NLLLoss() 

CrossEntropyLoss等价于输入先做log_softmax，再计算NLL损失

输入是一个对数概率向量和一个目标标签