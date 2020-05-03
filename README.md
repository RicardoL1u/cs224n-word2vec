# cs224n-word2vec




---

>谁又还不是一条直线的投影呢

##从one-hot开始说起
假设我们永远一个含有N个词的词典dictionary，它包含我们所拥有的语料库corpus中所有的单词，很自然的，我们可以想到用一个向量$X_i$,其中
$$
x_{ind} = \left\{ 
\begin{array}{ll} 1 & ind = i\\ 
0 & ind \not= i  \end{array} \right.
$$
>note:
在线性代数中，我们常常更倾向于使用列向量而不是行向量，然而在计算机中，数组是以行优先的，所以如果不加以特殊说明的话，我们一般使用行向量

来表示在词典中字典序为$i$的一个单词，这样我们就能很容易的用向量来表示每一个单词。
然而对于任意的两个单词之间，比如$X_{good}$ 与 $X_{great}$,两者的余弦相似度，也就是两个向量之间的所夹角的余弦值
$$\frac{X_{good} X_{great}^T}{|X_{good}||{X_{great}}|} = 0$$
这着实无法确切的描述两个单词之间的相似性

##Representing words by their context
就像英语阅读理解时的一样，我们总是可以从上下文中去猜测一个单词的semantic语言
distributional semantics语义: A word's meaning is given by the words that frequently appear close-by
> "You shall know a word by the company it keeps"       (J. R. Firth 1957: 11)

然后再用一个D-dim的向量去表示他。
在这种思想下的指导下，word2vec便诞生了
上下文----> window
猜测  ----> $P(w_{t+j}|w_t)$
取语料库corpus中位置为$t$的词为例，window size 设置为 2
![image.png](https://i.loli.net/2020/05/02/h3n2RGkxKWf5utq.png)
For each position $𝑡=1,…,𝑇, predict context words within a window of fixed size m, given center word $w_j$, 将位置t遍历整个语料库，假设每一个词之间都是independent的（便于各个似然概率之间直接相乘），我们就得到了目标函数objective function(AKA loss function)
![image.png](https://i.loli.net/2020/05/02/ixLzIG2Pf57rTJX.png)

>note
>需要注意的是，这里我们对于单词的下标又变成了$t,1\leq t\leq T$,而之前再one-hot向量中提到的的下标$ind,0\leq ind \leq N$是截然不同的两个概念，你可以理解为，在一本有30w词的小说中，假如说把这本小书作为我们的语料库corpus 位置position $t$  此时便从1~30w,$T = 30w$ , 而在这本书中所有出现过的不同的单词一共有N个，这N个词的集合set，便是我们的字典dictionary，也就是说我们的dictionary是有一个有$N$个元素的set *$\bf V$*

Qustion: How to calculate the $P(o|c)$
we use the softmax
![predictfun.png](https://i.loli.net/2020/05/02/bkWEnABuGtdKgiD.png)
>note: 这里的集合$V$就是我们所说dictionary了
在知道predication function之后，我们就得到了我们的objective function $J_{\text{skip gram}}$ 的完整形式，接下来所需要做的仅仅只不过是不断的优化参数$\theta$,使得loss不断下降。
 
## 如何优化$\theta$ ---->SDG Stochastic Gradient Descent
对于原始的objective function
$$J(\theta) = -\frac{1}{T}\sum_{t=1}^T\sum_{-m\leq j\leq m, j\not=0}logP(w_{t+j}|w_t;\theta)$$
**其中第一个求和符号**从$t=1\dots T$
而一般来说我们的$T$都是一个近千万左右的数字，对于计算的开销十分巨大，所以我们每次随机从$t=1\dots T$中随机抽取一些数字，来代替从$t=1\dots T$的求和
这就是随机梯度下降中“随机”了，设我们随机抽取的数字构成的集合为${\bf S}, length({\bf S}) < T$
我们可以将每一次迭代中的objective function改写为
$$J(\theta) = -\frac{1}{T}\sum_{t\in{\bf S}}\sum_{-m\leq j\leq m, j\not=0}logP(w_{t+j}|w_t;\theta)$$
假如我们对于每一个suppose the center word $c = w_t$,a single pair words $c$ and $o$, the loss is given by
$$J_{naive-softmax}(v_c,o,U)=-logP(O=o|C=c) = -u_o^Tv_c + log\sum_{w\in Vocb}exp(u_w^Tv_c)$$

这样我们就可以把
$$J_{skip-gram}(v_c,w_{t-m},\dots w_{t+m},{\bf U}) = \sum_{-m\leq j \leq m,j\not= 0}J_{naive-softmax}(v_c,w_{t+j},U)$$

首先我们对$J_{naive-softmax}(v_c,o,U)$求梯度
由于我们这里是从center word $v_c$来预测outside word $o$, 假设我们记 the true distibution $y$, and the prediction distribution $\hat{y}$
$$
\begin{eqnarray*}
\frac{\partial J }{\partial v_c} &=& -u_o + \sum_{w\in Vocb}P(O=w|C=c)u_w\\
&=& -u_o+(u_1,u_2,\dots,u_N)(y_1,y_2,\dots,y_N)^T\\
&=& U(\hat{y}-y)
\end{eqnarray*}
$$
>note:
在nlp中，下标总是一个非常容易让人疑惑的点，我们这里$y,\hat{y}$都是一个长度为$N$列向量，而其中，$y$是一个典型的one-hot向量，$y_i = 1, word_i = word_o$时，$otherwise, y_i = 0$,而显然的这里的长度$N$就是取决于此时语料库所对应的词典的大小$N$$

接着我们再考虑对$u_w$求偏导
当$w=o$
$$
\begin{eqnarray*}
\frac{\partial J }{\partial u_o} &=& -v_c + \frac{exp(u_o^Tv_c)v_c}{\sum_{w\in Vocb}exp(u_w^Tv_c)}\\
&=& -v_c + P(O=o|C=c)v_c\\
&=& v_c(-1+\hat{y_o})
\end{eqnarray*}
$$当$w\not=o$
$$
\begin{eqnarray*}
\frac{\partial J }{\partial u_o} &=& \frac{exp(u_o^Tv_c)v_c}{\sum_{w\in Vocb}exp(u_w^Tv_c)}\\
&=& P(O=w|C=c)v_c\\
&=& \hat{y_w}v_c
\end{eqnarray*}
$$
这样我们就得到了有关于$J_{naive-softmax}(v_c,o,U)$的全部梯度了

接着我们在基于$J_{naive-softmax}(v_c,o,U)$的梯度上
去计算$J_{skip-gram}(v_c,w_{t-m},\dots w_{t+m},{\bf U})$的梯度
$$\frac{\partial{J_{skip-gram}(v_c,w_{t-m},\dots w_{t+m},{\bf U}}) }{\partial{{\bf U}}} = \sum_{-m\leq j \leq m,j\not= 0}\partial{J_{naive-softmax}(v_c,w_{t+j},U)}/\partial{u_w}$$
>note:这里的意思是的，没有被窗口包括的outside word 的 $u_w$ 的梯度为0，这里也不太算是求和，实际上是一列列的列向量的并列

$$\frac{\partial{J_{skip-gram}(v_c,w_{t-m},\dots w_{t+m},{\bf U}}) }{\partial{{v_c}}} = \sum_{-m\leq j \leq m,j\not= 0}\partial{J_{naive-softmax}(v_c,w_{t+j},U)}/\partial{v_c}$$

when $w\not= c$
$$\frac{\partial{J_{skip-gram}(v_c,w_{t-m},\dots w_{t+m},{\bf U}}) }{\partial{{v_w}}} = 0$$

如此我们就获得word2vec过程中所需要的所有的梯度！！接下来我们就可以去实现word2vec了
![image.png](https://i.loli.net/2020/05/03/l6zqLaPpfwhEC3u.png)
>note:
需要注意到的是，在计算$\frac{\partial J_{naive-softmax} }{\partial v_c} $中，我们需要知道dictionary中全体word的$P(O=w|C=c)$的概率并求和，这实际上是一个非常大的计算开销，所以在接下来我们会提出neg-sampling 与 Hierarchical softmax两种新的求近似$P(O=w|C=c)$似然概率的方法，来代替$J_{naive-softmax}$，从而降低计算开销
