# **Transformer-based DKN for News Recommendation**

使用transformer改进DKN

![截屏2021-05-20 上午11.21.53](https://i.loli.net/2021/05/20/1udclXOfk7hS4gt.png)



DKN将新闻标题的单词与知识库的实体进行链接得到该entity对应的邻居entity，称为context entiltles，通过word2vec得到新闻标题序列的词嵌入，通过知识图嵌入模型得到knowledge entity embedding。平均context entiltles得到Context embedding

DKN通过非线性变换将entity embedding 和context embedding映射到同一个向量空间。然后使用单词嵌入、实体嵌入、上下文嵌入作为多通道CNN输入，通过卷积运算得到新闻标题的新闻嵌入。





![截屏2021-05-20 上午11.28.10](https://i.loli.net/2021/05/20/C1yTzMIFGP975iv.png)