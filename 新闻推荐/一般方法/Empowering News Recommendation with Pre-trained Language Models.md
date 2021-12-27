# Empowering News Recommendation with Pre-trained Language Models

**Chuhan Wu**, Fangzhao Wu, Tao Qi, Yongfeng Huang

SIGIR 2021 (Short Paper)

首次在新闻推荐任务使用预训练模型。



## 普通的新闻推荐框架

![截屏2021-04-28 下午8.55.07](https://i.loli.net/2021/04/28/3vumDe5arZdoFUi.png)



## PLM Empoweed News Recommendation

## ![截屏2021-04-28 下午8.58.03](https://i.loli.net/2021/04/28/uJrUeXlL1akbB64.png)

输入的新闻文本被分为M个token，PLM将每个token转变为向量，再经过attention统一为新闻的嵌入表示



## EXPERIMENTS

推荐模型：EBNR、NAML、NPA、NRMS

PLM：BERT、RoBERTa、UnilM

![截屏2021-04-28 下午9.09.31](/Users/mac/Library/Application Support/typora-user-images/截屏2021-04-28 下午9.09.31.png)



![截屏2021-04-28 下午9.18.35](https://i.loli.net/2021/04/28/dUSgt4OIrZ5Bf8A.png)



bert的规模越大，模型的效果越好

![截屏2021-04-28 下午9.20.04](https://i.loli.net/2021/04/28/3neOuSZVkQXF67p.png)



使用attention得到新闻特征的汇总嵌入表示效果最佳



![截屏2021-04-28 下午9.22.18](https://i.loli.net/2021/04/28/UujgrmfxQMcXE4b.png)

使用PLM比浅层模型更难区分新闻文本的情感特征，有利于提高新闻推荐的性能

## 总结

预训练模型如bert之类的确实能提高新闻推荐的性能，但一般只能提升1～3个点。