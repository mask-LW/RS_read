# Personalized News Recommendation using  Classified Keywords to Capture User Preference

Kyo-Joong Oh*, Won-Jo Lee*, Chae-Gyun Lim**, Ho-Jin Choi

## 一、

**新闻推荐的特点**：

1.个性化新闻推荐系统需要了解用户的长期兴趣、短期偏好、业务和情境背景、社会关系

2.构建推荐系统时，新闻项目与其他项目有区别，比如协同过滤可能不会起作用，因为新闻很快就失去价值，不值得被推荐。

3.热点新闻话题变化频繁。



**针对以上特点，作者提出了一种神经网络模型来分析用户对新闻推荐的偏好：**

该模型从特定用户过去阅读的新闻文章集合中提取兴趣关键字来描述用户偏好。对于关键词分类，我们使用深度神经网络进行在线偏好分析，因为需要自适应学习来敏感地跟踪热点的变化



目前新闻分析中常用的文献分析技术：

1.使用术语提取技术从文章中检测新闻事件

【Y. Yang, J. G. Carbonell, R. D. Brown, T. Pierce, B. T. Archibald, and X. Liu, “Learning approaches for detecting and tracking news events,” IEEE Intelligent Systems, vol. 14, no. 4, pp. 32-43, Jul.-Aug. 1999.】

2.文档聚类

【V. Hatzivassiloglou, L. Gravano, and A. Maganti. “An investigation of linguistic features and clustering algorithms for topical document clustering,” Proceedings of the 23rd Annual International ACM SIGIR Conference on Research and Development in Information Retrieval (ACM SIGIR 2000), pp. 224-231, Athens, Greece, July 24-28, 2000.】

3.文档索引

【K. M. Hammouda and M. S. Kamel, “Efficient phrase-based document indexing for web document clustering,” IEEE Transactions on Knowledge and Data Engineering, vol.16, no.10, pp. 1279-1296, Oct. 2004.】

4.文档摘要

【K. R. McKeown, R. Barzilay, D. Evans, V. Hatzivassiloglou, J. L. Klavans, A. Nenkova, and S. Sigelman, “Tracking and summarizing news on a daily basis with Columbia's Newsblaster,” Proceedings of the Second International Conference on Human Language Technology Research,, pp. 280-285, San Francisco, CA, USA, 2002】

5.情感分析经常被用来分析社交网络

【A. Balahur, R. Steinberger, M. A. Kabadjov, V. Zavarella, E. van der Goot, M. Halkia, B. Pouliquen, and J. Belyaeva, “Sentiment Analysis in the News,” Proceedings of the 7th International Conference on Language Resources and Evaluation (LREC 2010), pp. 2216-2220, Valletta, Malta, May 2010.】



关键字分类常用方法：

SVM、neural network、利用词频、文档逆频率等传统信息检索技术（TF-IDF）、基于统计方法的动态主题模型



**新闻分析中本文选择的是使用术语提取技术，从用户所阅读的新闻文章中识别出重要的关键词，从而获取用户偏好**

**关键词分类中本文选择使用DNN**，如限制玻尔兹曼机(RBM)和深度信念网络(DBN)





## 二、作者提出的方案

主要由**偏好分析**和**新闻推荐**组成

![image-20200916201957713](https://i.loli.net/2020/09/16/2Hka7v5cQAgnwXO.png)

### A.User Profiling Phase

从用户阅读的文章中分析用户感兴趣的关键词，根据分析结果，从而推荐其他个性化的新闻。

通过深度神经网络对兴趣关键词进行分类，每个词分为关键词和非关键词。

![image-20200916202735298](https://i.loli.net/2020/09/16/WJgNkOotUK4b3MY.png)

模型基于深度信念网络(DBN)进行修改，选择3层感知器，有5个输入，得到n个关键词作为输出

TF：单词在给定文档中出现的频率

ITF：文档中该单词的总频率的反置值

T：单词在文档标题中是否出现

FP：单词在文档第一句中是否出现

CP：按关键词分类的长期兴趣权重

最后得到用户的偏好关键词列表，作为user profile为下一阶段做准备。

### B.News Ranking Phase 

![image-20200916204040535](https://i.loli.net/2020/09/16/iHJ7ZQS9Oy1kDxz.png)

由A阶段得到User Profile后，首先对即将发布的新闻内容进行过滤后提取名词，然后计算该新闻中的各个名词的TF/IDF分数，最后通过余弦相似度来计算User Profile和News Profile的相似度得分，从而得到为用户推荐的top-k新闻。



## 三、实验

作者实现一个新闻推荐服务的原型，对一群用户进行简单研究。

### A.Data Collection and Pre-processing

从 Google  Chrome Extension  “Daum  News  Tracker”  和   Android  application“KECI  News.” 收集数据，两个月用于训练，三个月用于测试



![image-20200916204954920](/Users/mac/Library/Application Support/typora-user-images/image-20200916204954920.png)

### B.Keyword Classification and User Profiling

使用了大约70个用户的1500篇文章数据来进行训练，数据是2013年3月至4月获得的，使用TF-IDF前50%的结果作为关键词标签，制作了包含兴趣关键词和关键词偏好权重的user profile.

![image-20200916210021588](/Users/mac/Library/Application Support/typora-user-images/image-20200916210021588.png)

相比自适应的IF-IDF，作者提出的模型训练时间增加，但accuracy增加程度更多。

### C.Personalized News Recommendation

本文使用基于内容的推荐

首先通过新闻爬虫收集每天的最新新闻集，进行预处理，最后得到News Profile，最后通过余弦相似度计算为用户推荐新闻

![image-20200916210558858](https://i.loli.net/2020/09/16/DWG7deYRtscp1lE.png)

8个用户5天内的推荐结果的命中率的平均值为54%，高于随机推荐新闻37%。

## 四、总结

文章提出的是对应用户偏好分析的模型，再结合基于内容的推荐来实现对新闻的个性化推荐。