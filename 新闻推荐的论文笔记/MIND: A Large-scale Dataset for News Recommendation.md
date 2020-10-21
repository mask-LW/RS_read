# MIND: A Large-scale Dataset for News Recommendation

MIND由用户点击微软新闻的日志构建而成，拥有100万用户和160000多篇英语新闻文章，每篇文章都有丰富的文本内容，如标题、摘要和正文

## 目前存在的新闻推荐的数据集

![截屏2020-10-13 下午8.41.20](/Users/mac/Library/Application Support/typora-user-images/截屏2020-10-13 下午8.41.20.png)

Plista4数据集：包含70,353篇新闻文章和1,095,323次点击事件。该数据集中的新闻文章是德语，用户主要来自德语世界。

【链接：http://www.newsreelchallenge.org/dataset/】

Adressa：（挪威语）48486篇新闻文章，3083438个用户和27223576个点击事件。

每个点击事件包含几个特性，如会话时间、新闻标题、新闻类别和用户ID。

每个新闻文章都与一些详细信息相关联，如作者、实体和主体。

【链接：http://reclab.idi.ntnu.no/dataset/】

Globo：（巴西语）包含大约314,000个用户，46,000篇新闻文章和300万次点击记录。

每个单击记录都包含用户ID、新闻ID和会话时间等字段

每一篇新闻文章都有ID、类别、发布者、创建时间，以及由预先在新闻元数据分类任务上训练过的神经模型生成的单词嵌入，但不提供新闻文章的原文。

【链接：https://www.kaggle.com/gspmoreira/news-portal-user-interactions-by-globocom】

Yahoo！：包含14,180篇新闻文章和34,022个点击事件。每一篇新闻文章都用文字ID表示，不提供原始的新闻文本。

【链接：https://webscope.sandbox.yahoo.com/catalog.php?datatype=1】

以上皆比较难用。

## MIND

从Microsoft News的用户行为日志中随机抽取了100万名用户，他们在2019年10月12日至11月22日的6周内至少有5次新闻点击记录.

最后一周用于测试，第五周用于训练，前四周的点击行为来构建新闻点击历史记录，第五周最后一天的样本作为验证集。

为了保护用户隐私，使用一次性的salt映射将每个用户安全地散列到一个匿名ID中，然后将其从生产系统中分离出来。我们收集了这些用户在此期间的行为日志，并将其整理成印象日志。

印象日志记录用户在特定时间访问新闻网站主页时显示给用户的新闻文章，以及用户对这些新闻文章的点击行为，通过添加新闻点击日志来构造用户的历史印象标签样本进行训练和检验新闻推荐模型,其格式为：

【uID, t, ClickHist, ImpLog】

MIND数据集中的每一篇新闻文章都包含一个新闻ID、一个标题、一个摘要、一个正文和一个类别标签，比如由编辑手动标记的“Sports”

为了便于知识感知型新闻推荐的搜索，我们将新闻文章的标题、摘要和正文中的实体提取到MIND dataset中，并使用内部的NER和实体链接工具链接到WikiData中的实体。还从WikiData中提取了这些实体的知识三元组，并使用TransE方法来学习实体和关系的嵌入。这些实体、知识三元组以及实体和关系嵌入也包含在数据集中

【此步骤应该是为了论文DKN使用】



### MIND细节

![截屏2020-10-13 下午9.39.58](/Users/mac/Library/Application Support/typora-user-images/截屏2020-10-13 下午9.39.58.png)

![截屏2020-10-13 下午9.41.14](/Users/mac/Library/Application Support/typora-user-images/截屏2020-10-13 下午9.41.14.png)

Survival TIme 的单位为天。大多数新闻存活时间不超过两天。**导致新闻的冷启动问题**

### Dataset Format

训练和验证集都是一个zip压缩文件夹，其中包含四个不同的文件：

| File Name              | Description                        |
| ---------------------- | ---------------------------------- |
| behaviors.tsv          | 用户的点击历史和印象日志           |
| news.tsv               | 新闻文章的信息                     |
| entity_embedding.vec   | 从知识图提取的新闻中的实体嵌入     |
| relation_embedding.vec | 从知识图提取的实体之间的关系的嵌入 |

#### behaviors.tsv

- Impression ID. The ID of an impression.
- User ID. The anonymous ID of a user.
- Time. The impression time with format "MM/DD/YYYY HH:MM:SS AM/PM".
- History. The news click history (ID list of clicked news) of this user before this impression. The clicked news articles are ordered by time.
- Impressions. List of news displayed in this impression and user's click behaviors on them (1 for click and 0 for non-click). The orders of news in a impressions have been shuffled.

| Column        | Content                                                      |
| ------------- | ------------------------------------------------------------ |
| Impression ID | 91                                                           |
| User ID       | U397059                                                      |
| Time          | 11/15/2019 10:22:32 AM                                       |
| History       | N106403 N71977 N97080 N102132 N97212 N121652                 |
| Impressions   | N129416-0 N26703-1 N120089-1 N53018-0 N89764-0 N91737-0 N29160-0 |

#### news.tsv

- News ID
- Category
- SubCategory
- Title
- Abstract
- URL
- Title Entities (entities contained in the title of this news)
- Abstract Entities (entites contained in the abstract of this news)

| Column           | Content                                                      |
| ---------------- | ------------------------------------------------------------ |
| News ID          | N37378                                                       |
| Category         | sports                                                       |
| SubCategory      | golf                                                         |
| Title            | PGA Tour winners                                             |
| Abstract         | A gallery of recent winners on the PGA Tour.                 |
| URL              | https://www.msn.com/en-us/sports/golf/pga-tour-winners/ss-AAjnQjj?ocid=chopendata |
| Title Entities   | [{"Label": "PGA Tour", "Type": "O", "WikidataId": "Q910409", "Confidence": 1.0, "OccurrenceOffsets": [0], "SurfaceForms": ["PGA Tour"]}] |
| Abstract Entites | [{"Label": "PGA Tour", "Type": "O", "WikidataId": "Q910409", "Confidence": 1.0, "OccurrenceOffsets": [35], "SurfaceForms": ["PGA Tour"]}] |

##### Title Entities

| Keys              | Description                                                  |
| ----------------- | ------------------------------------------------------------ |
| Label             | The entity name in the Wikidata knwoledge graph              |
| Type              | The type of this entity in Wikidata                          |
| WikidataId        | The entity ID in Wikidata                                    |
| Confidence        | The confidence of entity linking                             |
| OccurrenceOffsets | The character-level entity offset in the text of title or abstract |
| SurfaceForms      | The raw entity names in the original text                    |

#### entity_embedding.vec和relation_embedding.vec

Entity_embedding.vec和Relation_embedding.vec文件包含通过TransE方法从子图（从WikiData知识图）获知的实体和关系的100维嵌入。在两个文件中，第一列是实体/关系的ID，其他列是嵌入矢量值。

## 实验方法

### 传统的推荐方法

LibFM(Rendle, 2012)，一种基于因子分解机的经典推荐方法。除了用户ID和新闻ID，我们还使用从先前单击的newand candidate news中提取的contentfeatures13作为附加特性来表示用户和候选新闻。

DSSM(Huang et al.， 2013)，深度结构化语义模型，使用三元哈希和多前馈神经网络进行查询文档匹配。我们使用从先前点击的新闻中提取的内容特性作为查询，从候选新闻中提取的内容特性作为文档

Wide&Deep(Cheng et al.， 2016)是一种双通道神经推荐方法，它有一个宽线性变换通道和一个深度神经网络通道。我们为两个频道使用相同的用户和候选新闻内容特性。

DeepFM(Guo et al.， 2017)，另一种流行的神经推荐方法，它综合了深度神经网络和因子分解机器。用户和候选新闻的相同内容特性被提供给两个组件

### 新闻推荐方法

DFM：深度融合模型(deep fusion model)是一种新闻推荐方法，它使用一种嵌入网络，将不同深度的神经网络结合起来，捕捉特征之间的复杂交互。

GRU：一种神经新闻推荐方法，利用自动编码器从新闻内容中学习潜在的新闻表示，并利用GRU网络从点击的新闻序列中学习用户表示

DKN：一种知识感知的新闻推荐方法，它使用CNN从包含单词嵌入和实体嵌入(从知识图推断)的新闻标题中学习新闻表示，并根据候选新闻和之前点击新闻的相似度来学习用户表示

NPA：一种基于个性化注意机制的神经新闻推荐方法，根据用户偏好选择重要词汇和新闻文章，以获取更多的信息和用户表现形式

NAML：一种专注多视角学习的神经新闻推荐方法，将不同类型的新闻信息融入到新闻文章的表现中

LSTUR：一种具有长期和短期用户兴趣的神经新闻推荐方法。它利用GRU从用户最近点击的新闻中塑造短期用户兴趣，从整个点击历史中塑造长期用户兴趣

NRMS：一种神经新闻推荐方法，利用多头自我注意从新闻文本中的单词中学习新闻表示，从先前点击的新闻文章中学习用户表示（使用技术较新）

### 实验结果

![截屏2020-10-14 上午11.20.11](/Users/mac/Library/Application Support/typora-user-images/截屏2020-10-14 上午11.20.11.png)

1⃣️新闻推荐方法一般比传统推荐方法要好。

在这些特定于新闻的推荐方法中，新闻文章和用户兴趣的表示是以端到端的方式学习的，而在一般的推荐方法中，它们通常使用手工制作的特性来表示。但DFM除外。

【使用神经网络从原始数据中学习新闻内容和用户兴趣的表示比特征工程更有效】

2⃣️NRMS效果最佳，表明先进的NLP技术对应新闻推荐能够有效地提高对新闻内容的理解和对用户兴趣的建模

3⃣️在AUC度量上，新闻推荐方法对未见用户的性能略低于对训练数据中包含的重叠用户的性能。然而，两种用户在mrr和nDCG指标上的性能没有显著差异。这一结果表明，通过对用户先前点击的新闻内容的推断，可以有效地将部分用户训练的新闻推荐模型应用于剩余用户和未来的新用户

### News Content Understanding

如何从文本内容中学习准确的新闻表述。











