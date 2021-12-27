# NewsBERT: Distilling Pre-trained Language Model for Intelligent News Application

**[Chuhan Wu](https://arxiv.org/search/cs?searchtype=author&query=Wu%2C+C)**, [Fangzhao Wu](https://arxiv.org/search/cs?searchtype=author&query=Wu%2C+F), [Tao Qi](https://arxiv.org/search/cs?searchtype=author&query=Qi%2C+T), [Yongfeng Huang](https://arxiv.org/search/cs?searchtype=author&query=Huang%2C+Y)

In Proceedings of ACM SIGKDD Conference on Knowl- edge Discovery and Data Mining (KDD 2021), Jennifer B. Sartor and Theo D’Hondt (Eds.). ACM, New York, NY, USA, Article 4, 9 pages. https://doi. org/10.475/123_4



## 前言

PLM规模太大，而新闻推荐需要低延迟容忍、且现在PLM是基于通用语料训练的，未必适用于新闻推荐领域，因此提出：1⃣️NewSBERT，对PLM进行蒸馏，用于新闻应用，包括新闻主题分类、假新闻检测、新闻标题生成、新闻检索、个性化新闻推荐等

2⃣️教师-学生联合学习和提炼框架来协作学习教师和学生模型。

3⃣️动量蒸馏方法，将教师模型的梯度加入到学生模型的更新中，以更好地传递教师模型学到的有用知识。





## METHODOLOGY

![截屏2021-04-28 下午9.56.29](/Users/mac/Library/Application Support/typora-user-images/截屏2021-04-28 下午9.56.29.png)

以上为一个典型的新闻分类任务中的总体框架。

teacher模型和student模型联合训练，同时从teacher模型蒸馏知识。

teacher为embedding层+ NK个transformer层

student为embedding层+ K个transformer层，比teacher模型要快k倍。

新闻文本经过transformer得到token，使用attentionon pooling分别得到teacher和stduent的新闻特征表示，经过一个share Dense来预测文本分类。

通过共享pooling层和dense的参数，学生模式可以从教师那里获得更丰富的监督信息，教师也可以了解学生的学习状况。因此，教师和学生可以互相学习，分享他们编码的有用知识，这有助于学习一个强大的学生模式。









![截屏2021-04-28 下午10.14.53](https://i.loli.net/2021/04/28/1VIc2xZyaY9vqXi.png)















## 总结

针对新闻应用的模型和知识蒸馏联合使用，核心是知识蒸馏