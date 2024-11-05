##### 1.LSTM
- 为了解决RNN的梯度消失以及捕捉长期依赖能力的缺点
- 核心思想是细胞状态以及三个门来控制细胞状态的更新
- 对于每个LSTM单元，有三个输入，分别是Ct-1，ht-1，xt，输出两个，分别是Ct，ht，有三个门，分别是遗忘门，输入门和输出门
	- ![[1707967764077.png]]
	- $$\begin{aligned}
		f_{t} = \sigma \left ( W_{f}\cdot\left [ h_{t-1},x_{t}   \right ] +b_{f}    \right)
		\end{aligned}$$
		LSTM 的第一步就是决定细胞状态需要丢弃哪些信息，这部分操作是通过一个称为忘记门的 sigmoid 单元来处理的，0 表示不保留，1 表示都保留
	- ![[1707967778812.png]]
	- $$\begin{aligned}
		i_{t} & =\sigma\left(W_{i} \cdot\left[h_{t-1}, x_{t}\right]+b_{i}\right)\\
		\tilde{C}_{t} & =\tanh \left(W_{C} \cdot\left[h_{t-1}, x_{t}\right]+b_{C}\right)
		\end{aligned}$$
		下一步是决定给细胞状态添加哪些新的信息。这一步又分为两个步骤，首先，利用输入门来决定更新哪些信息。然后利用通过一个 tanh 层得到新的候选细胞信息，这些信息可能会被更新到细胞信息中。
	- ![[1707967784362.png]]
	- $$\begin{aligned}
		C_{t}=f_{t} * C_{t-1}+i_{t} * \tilde{C}_{t}
		\end{aligned}$$
		下面将旧的细胞信息和新的候选细胞信息利用之前的遗忘门和输入门更新为新的细胞信息
	- ![[1707967790503.png]]
	- $$\begin{aligned}
		o_{t} & =\sigma\left(W_{o}\left[h_{t-1}, x_{t}\right]+b_{o}\right) \\
		h_{t} & =o_{t} * \tanh \left(C_{t}\right)
		\end{aligned}$$
		更新完细胞状态后进行更新ht，通过一个输出门来控制新的细胞信息的多少给ht
##### 2.GRU
- ![[1707969772913.png]]
- $$\begin{aligned}
		z_{t} & =\sigma\left(W_{z} \cdot\left[h_{t-1}, x_{t}\right]\right) \\
		r_{t} & =\sigma\left(W_{r} \cdot\left[h_{t-1}, x_{t}\right]\right) \\
		\tilde{h}_{t} & =\tanh \left(W \cdot\left[r_{t} * h_{t-1}, x_{t}\right]\right) \\
		h_{t} & =\left(1-z_{t}\right) * h_{t-1}+z_{t} * \tilde{h}_{t}
		\end{aligned}$$
		它将忘记门和输入门合并成一个新的门，称为更新门。GRU 还有一个门称为重置门
##### 3.Word2Vec
- 分为cbow和skip-gram两种
	-  cbow 是用周围词预测中心词，从而利用中心词的预测结果情况，使用梯度下降方法，不断的去调整周围词的向量。当训练完成之后，每个词都会作为中心词，把周围词的词向量进行了调整，这样也就获得了整个文本里面所有词的词向量
	- skip-gram 是用中心词来预测周围的词，从而利用周围的词的预测结果情况，使用梯度下降来不断的调整中心词的词向量，最终所有的文本遍历完毕之后，也就得到了文本所有词的词向量。
- 优缺点
	- skip-gram 出来的准确率比 cbow 高，但训练时间要比 cbow 要长
	- skip-gram 在预测生僻字的预测效果更好
	- 原因是因为skip-gram相对于cbow来说，训练样本更多
- 损失函数
![[1718504830914.png]]
- 负样本选择
	- 负样本是从词汇表中随机选择的，但选择的概率并非均匀。
![[1718504883781.png]]
##### 4.Transformer
- 优势
	- 并行计算：Transformer 中的自注意力机制使得每个位置的表示都可以与其他位置并行计算，不像 RNN 那样需要按顺序逐步计算。这使得 Transformer 在处理长序列时具有更高的计算效率。
	- 长期依赖关系：由于自注意力机制可以计算序列中任意两个位置之间的相关性，Transformer 能够更好地捕捉长期依赖关系。相比之下，RNN 模型在处理长序列时容易出现梯度消失或梯度爆炸的问题。
	- 捕捉全局信息：传统的循环神经网络在处理长序列时会逐步积累上下文信息，导致较早的输入信息逐渐被遗忘。而 Transformer 通过自注意力机制，可以同时考虑整个序列的信息，能够更好地捕捉全局上下文信息。
- 为什么要除以根号dk
	- **缓解点积量级大的问题**：当dk​ 较大时，点积的结果也会变得较大，容易产生数值稳定性问题，比如溢出，影响精度
	- **保持训练稳定**：当dk​ 较大时，点积的结果也会变得较大，那么经过softmax后，输出变得极端，某些趋近于1，某些趋近于0，而位于这些位置时的梯度是0，从而产生梯度消失
	- **数学上的合理性**：从数学角度来看，假设 Q 和 K 中的元素是均值为 0，方差为 1 的独立同分布随机变量，向量的维度为 dk，内积的期望为0，方差为dk，因此除以根号dk会将内积的方差重新变成1，数值更加稳定，梯度传播更加稳定
- cos位置编码缺点
	- 缺乏适应性，正弦余弦位置编码是预先定义的，它不是模型训练过程中学习得到的。这种固定的编码方式可能不如可学习的位置编码（例如，将位置编码作为模型参数来学习）那样灵活，可能在某些情况下不是最优的表示方式。
	- 缺乏绝对位置信息，相对位置偏重：虽然正弦余弦编码能够很好地捕获位置间的相对关系，但它可能不足以强调序列中单词的绝对位置信息。在某些任务中，绝对位置可能同样重要
	- 编码的周期性，周期性可能引入干扰：正弦余弦编码的周期性可能在某些情况下引入不必要的干扰，尤其是在序列非常长时，不同位置的编码可能会过于相似，从而影响模型的区分能力。
	- 如何改进
		- 可学习的位置嵌入：将位置嵌入作为模型参数的一部分，通过训练过程学习位置嵌入，而不是使用固定的正弦和余弦函数。
		- 加入绝对位置信息
- Pre_Norm vs Post_Norm
	- Output=LayerNorm(x+SubLayer(x))
	- Output=x+LayerNorm(SubLayer(x))
	- Pre-Norm:
		- 优点: 更稳定的梯度传播，尤其是在深层网络中表现更好，可以缓解梯度消失和梯度爆炸问题。
		- 缺点: 可能在早期训练阶段收敛速度较慢，因为输入的归一化可能会影响模型捕捉复杂模式的能力。
	- Post-Norm:
		- 优点: 在浅层网络中效果较好，最早被 Transformer 模型采用。
		- 缺点: 在深层网络中容易出现梯度消失或梯度爆炸的问题，导致训练不稳定。
- 单头和多头参数量和计算量的区别
	- 参数量多头大，因为多出了最后线性变换部分
	- 计算量多头大，因为多出了最后线性变换部分
##### 5.手撕单注意力
```
from math import sqrt
import torch
import torch.nn as nn

class ScaledDotProductAttention(nn.Module):
	def __init__(self, dim_in, dim_k, dim_v): 
		super().__init__() 
		self.dim_in = dim_in 
		self.dim_k = dim_k 
		self.dim_v = dim_v 
		self.linear_q = nn.Linear(dim_in, dim_k, bias=False) 
		self.linear_k = nn.Linear(dim_in, dim_k, bias=False) 
		self.linear_v = nn.Linear(dim_in, dim_v, bias=False) 
		self._norm_fact = 1 / sqrt(dim_k) 
	
	def forward(self, x):
		batch, n, dim_in = x.shape 
		q = self.linear_q(x) # batch, n, dim_k 
		k = self.linear_k(x) # batch, n, dim_k
		v = self.linear_v(x) # batch, n, dim_v 
		dist = torch.bmm(q, k.transpose(1, 2)) * self._norm_fact 
		dist = torch.softmax(dist, dim=-1)
		att = torch.bmm(dist, v) 
		return att
```
##### 6.手撕多注意力
```
from math import sqrt
import torch
import torch.nn as nn

class ScaledDotProductAttention(nn.Module):
	def __init__(self, dim_in, dim_k, dim_v, num_heads=8): 
		super().__init__() 
		self.dim_in = dim_in 
		self.dim_k = dim_k 
		self.dim_v = dim_v 
		self.num_heads = num_heads
		self.linear_q = nn.Linear(dim_in, dim_k, bias=False) 
		self.linear_k = nn.Linear(dim_in, dim_k, bias=False) 
		self.linear_v = nn.Linear(dim_in, dim_v, bias=False) 
		self._norm_fact = 1 / sqrt(dim_k // self.num_heads) 
		self.fc = nn.Linear(dim_v, dim_v) 
	
	def forward(self, x):
		batch, n, dim_in = x.shape 
		dk = self.dim_k // self.num_heads
		dv = self.dim_v // self.num_heads
		
		q = self.linear_q(x).reshape(batch, n, self.num_heads, dk).transpose(1, 2) # (batch, nh, n, dk) 
		k = self.linear_k(x).reshape(batch, n, self.num_heads, dk).transpose(1, 2) # (batch, nh, n, dk) 
		v = self.linear_v(x).reshape(batch, n, self.num_heads, dv).transpose(1, 2) # (batch, nh, n, dk) 
		dist = torch.bmm(q, k.transpose(2, 3)) * self._norm_fact # (batch, nh, n, n)
		dist = torch.softmax(dist, dim=-1)
		att = torch.bmm(dist, v) # (batch, nh, n, dv)
		att = att.transpose(1, 2).reshape(batch, n, self.dim_v)  # batch, n, dim_v
		att = self.fc(att) # batch, n, dim_v
		return att
```