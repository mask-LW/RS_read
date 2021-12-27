# Neural News Recommendation with Long- and Short-termUser Representations

Mingxiao An,University of Science and Technology of China, Hefei 230026, China

Fangzhao Wu, Microsoft Research Asia, Beijing 100080, China

Chuhan Wu, Department of Electronic Engineering, Tsinghua University, Beijing 100084, China

Kun Zhang, University of Science and Technology of China, Hefei 230026, China

Zheng Liu, Microsoft Research Asia, Beijing 100080, China

Xing Xie，Microsoft Research Asia, Beijing 100080, China

**Publication:** Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 336–345 Florence, Italy, July 28 - August 2, 2019. c 2019 Association for Computational Linguistics

 Cited times: 23



**propose  a  neural  news  recommendation  approach  which  can  learn  both  long-and short-term user representations**

## LSTUR

### News Encoder：learn representations of news from their titles,  topic and subtopic categories

![截屏2020-10-17 下午9.01.47](https://i.loli.net/2020/10/17/3CLa68T7xg4EwcM.png)

####  title  encoder 

1⃣️Denote the word sequence in a news title t as $t= [w1,w2,...,wN]$，N is the length of the news title

via a word embedding matrix, transform into : $\color{black}{[w1,w2,...,wN]}$

2⃣️apply  a  CNN  network  to  learn  contextual  word representations by capturing the local context information.   

Denote  the  contextual  representation of wi as ci, which is computed as follows：
$$
c_i = RELU(C *w_{[i-M:i+M]}+b),
$$
$w_{[i-M:i+M]}$ is the concatenation of the embeddings  of  words  between  position $i−M$ and  $i+M$ , C and b are the parameters of the convolutional filters in CNN, and M is the window size

3⃣️employ a word-level attention networkto  select  important  words  in  news  titles  to  learnmore  informative  news  representations.   

The  attention weight αi of  the i-th word is formulatedas follows:
$$
a_i = tanh(v*c_i+b)
$$

$$
\alpha_i = \frac{exp(ai)}{∑_{j=1}^N=exp(a_j)}, (2)
$$

The  final  representation  of  a  news  titletis  the summation of its contextual word representations weighted by their attention weights as follows
$$
e_t=\sum^\N_{i=1}=α_ic_i,(3)
$$

#### topic encoder

learn the representations  of  topics  and  subtopics  from  the embeddings of their IDs, Denote $e_v$ and $e_{sv}$ as the representations of topic and subtopic.

#### Final

The final representation of a news article is the concatenation of the representations of its title, topic and subtopic, i.e.,$e= [e_t,e_v,e_{sv}]$

### User  Encoder : learn representations of users from the history of their browsed news

####  Short-Term User Representation: capture user’s temporal interests

Online users may have dynamic short-term interests in reading news articles, which may be influenced by specific contexts or temporal information demands

【阅读新闻时可能具有动态的短期兴趣，这可能受到特定语境或时间信息需求的影响】

We propose to learn the short-term representations of users from their recent browsing history to capturetheir  temporal  interests,  and  use  gated  recurrent networks (GRU)  network to capture the sequential news reading patterns

Denote  news  browsing  sequence from a user sorted by timestamp in ascending order as $C=\lbrace{c1,c2,...,ck}\rbrace$, where k is the length of this sequence

apply the news encoder to obtain  the  representations  of  these  browsed  articles, denoted as $\lbrace{e_1,e_2,...,e_k}\rbrace$.

The short-termuser representation is computed as follows:


$$
r_t = \sigma(W_r[h_{t-1},e_t])
$$

$$
z_t = \sigma(W_z[h_{t-1},e_t])
$$

$$
\tilde{h}_t = tanh(W_h[r_t\odot h_{t-1},e_t])
$$

$$
h_t = z_t \odot h_t +  (1-z_t) \odot \tilde{h}_t
$$

σ is  the  sigmoid  function, $$\odot$$ is  the  item-wise product(dot product)

The  short-term  user representation is the last hidden state of the GRU network, i.e., $u_s = h_k$​



####  Long-Term User Representations: capture user’s consistent preferences

the long-term user representations are learned from the embeddings of the user IDs, which are randomly initialized and fine-tuned during model training

Denoteuas the ID of a user and  $W_u$ as the look-up table for long-term user representation, the long-term user representation of this user is  $u_l=W_u[u]$

####  Long- and Short-Term UserRepresentation

two methods to combine the long-term and short-term user presentations  for  unified  user  representation

![截屏2020-10-17 下午9.57.29](https://i.loli.net/2020/10/18/otYP5NirzRWOlsk.png)

LSTUR-ini: using  the  long-term  user representation to initialize the hidden state of the GRU network in the short-term user representation model , and the last hidden state of the GRU network as the final user representation

LSTUR-con: concatenating the long-term user representation with the short-term user representation as the final user representation



### Model Training

Denotethe representation of a user u as $ u$  and the representation of a candidate news article ex as $e_x$, the probability score $s(u,c_x)$of this user clicking this news is computed as $s(u,c_x) =u^Tex$

For each news browsed  by  a  user  (regarded  as  a  positive  sample),  we randomly sample K news articles from the same impression which are not clicked by this user as negative sample.

the news click prediction problem is reformulated as a pseudo K+ 1-way classification task

minimize the summation of the negative log-likelihood of all positive samples during training, which canbe formulated as follows:
$$
-\sum^P_{i=1}log\frac{exp (s(u,c_i^p))}{exp (s(u,c_i^p)) + \sum^K_{k=1}exp(s(u,c_{i,k}^n))},(5)
$$
$P$ is the number of positive training samples,  and $c^n_{i,k}$ is  the k-th  negative  sample  in  the same session with the i-th positive sample.

Since not all users can be incorporated in news recommendation  model  training  (e.g.,  the  newcoming users), it is not appropriate to assume all users have long-term representations in our models in the prediction stage.

randomly mask the long-term representations of users with a certain probability $p$.

the long-term user representation inour LSTUR approach can be reformulated as:
$$
u_l = M \cdot W_u[u],M ~ B(1,1-p),(6)
$$

## Experiments

#### Dataset

MSN  News in  four  weeks from December 23rd, 2018 to January 19th, 2019.

We used the logs in the first three weeks for model training,  and those in the last week for test.  We also randomly sampled 10% of logs from the training  set  as  the  validation  data..

For  each  sample,we collected the browsing history in last 7 days tolearn short-term user representations

![截屏2020-10-18 下午3.22.44](https://i.loli.net/2020/10/18/9pRqGul7TStjLCx.png)

#### Result

![截屏2020-10-18 下午3.28.01](https://i.loli.net/2020/10/21/IYWFu4fpoEMXlNj.png)

#### Effectiveness of Long- and Short-Term User Representation

![截屏2020-10-18 下午3.43.23](https://i.loli.net/2020/10/21/wcEpKeZdHTFC5UI.png)

#### Effectiveness of News Encoders in STUR

![截屏2020-10-18 下午3.48.28](https://i.loli.net/2020/10/21/ft9dRSB1QeWhcGL.png)

We explore the effectiveness of GRU in en-coding news by replacing it with several other encoders, including:  1) Average:  using the averageof all the news representations in recent browsing history;

 2) Attention: the summation of news representations weighted by their attention weights;

3) LSTM, replacing GRU with LSTM. 

the sequence-based encoders can capture the sequential new reading patterns to learnshort-term representations of users, which is difficult for Average and Attention based encoders

GRU contains fewer parameters and has lower risk of overfitting than LSTM

#### Effectiveness of News Title Encoders

the  news  encoder  is  a  combination  of CNN network and an attention network (denoted as CNN+Att)

compare it with several variants, i.e., CNN, LSTM, and LSTM with attention(LSTM+Att):

![截屏2020-10-18 下午4.23.13](https://i.loli.net/2020/10/21/syF4wOxte8rP9nE.png)

using  CNN outperform those using LSTM may be because local contexts in news titles are more important for learning news representations.

#### Effectiveness of News Topic

![截屏2020-10-18 下午4.34.44](https://i.loli.net/2020/10/21/VYBbTCou74qZKLa.png)

incorporating  either  topics or subtopics can effectively improve the performance of our approach

####  Influence of Masking Probability

![截屏2020-10-18 下午4.41.52](https://i.loli.net/2020/10/21/XuKTkQHxwLRIZeW.png)

A moderate choice on $p$ (e.g.,0.5) is most appropriate for both LSTUR-ini andLSTUR-con methods, which can properly balancethe learning of LTUR and STUR