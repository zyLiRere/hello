# 深度学习入门

### 一、基础学习

---

需要预备的基础知识，主要的学习材料包括（仅供参考）。

1. **深度学习**
	- **深度学习课程 (by 虞剑飞)**
	- 神经网络与深度学习网络课程 (by 邱锡鹏) [视频1-6讲](https://www.bilibili.com/video/BV13b4y1177W) [视频7-10讲](https://www.bilibili.com/video/BV19u411d7r3)
	-  CS224n Natural Language Processing with Deep Learning [课程首页](http://web.stanford.edu/class/cs224n/) [视频](https://www.bilibili.com/video/BV18Y411p79k)
2. **自然语言处理**
	- CS124 From Languages to Information [课程首页](https://web.stanford.edu/class/cs124/)  [视频](https://www.bilibili.com/video/BV1kM4y1N7A2)
	- 自然语言处理 (by 李宏毅) [课程首页](https://speech.ee.ntu.edu.tw/~hylee/dlhlp/2020-spring.php)

### 二、书籍材料阅读

---

1. Zong, Chengqing,Rui Xia, and Jiajun Zhang. Text Data Mining. SpringerSingapore, 2021.
2. 宗成庆著.统计自然语言处理 (第2版)，清华大学出版社，2013.
3. [邱锡鹏著，神经网络与深度学习.机械工业出版社，2020.](https://nndl.github.io/)

### 三、入门复现

---

需要随课程完成的入门任务，主要与深度学习课程  (by 虞剑飞)同步进行。

#### 1. 基于卷积神经网络的目标检测任务
- **任务**：利用卷积神经网络，实现对MNIST数据集的分类问题。
- **数据集**：MNIST数据集包括 60000 张训练图片和 10000 张测试图片。图片样本的数量已经足够训练一个很复杂的模型（例如 CNN 的深层神经网络）。除此之外， MNIST 数据集是一个相对较小的数据集，可以在你的笔记本 CPUs 上面直接执行。 [下载链接](https://doc.codingdict.com/tensorflow/tfdoc/tutorials/mnist_download.html)
- **要求**：Tensorflow或者Pytorch实现（最好为Pytorch实现）
- **评价指标**：准确率
- **参考**：LeNet;AlexNet等
- 


#### 2. 基于卷积神经网络的文本情感分类
- **任务**：利用卷积神经网络，实现对一个电影评论数据集(Rotten Tomatoes dataset)的情感分类问题
- **数据集**：句子级情感五分类（very negative, negative, neutral, positive, very positive）[原始数据下载链接](https://nlp.stanford.edu/sentiment/sentiment/)
- **要求**：Tensorflow或者 Pytorch实现（最好为Pytorch实现）
- **评估指标**：准确率
- **参考论文**：[Convolutional Neural Networks for Sentence Classification](https://arxiv.org/pdf/1408.5882.pdf)

#### 3. 基于循环神经网络的命名实体识别
- **任务**：利用循环神经网络，识别出CoNLL-2003数据集的实体以及类别
- **数据集**：[CONLL-2003](https://www.clips.uantwerpen.be/conll2003/ner/)
- **要求**：Tensorflow或者 Pytorch实现（最好为Pytorch实现）
- **评估指标**：Precision、Recall、F1
- **参考资料**：
- 1. 《神经网络与深度学习》第6、11章
- 2. [End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF](https://arxiv.org/pdf/1603.01354.pdf)
- 3. [Neural Architectures for Named Entity Recognition](https://arxiv.org/pdf/1603.01360.pdf)

#### 4.基于层级式的注意力机制网络的文档分类
- **任务**：基于层级式的注意力机制网络，进行Yelp Review 2013数据集的文档分类
- **数据集**：[IMDB reviews](https://github.com/nihalb/JMARS)
- 
- **要求**：Tensorflow或者 Pytorch实现（最好为Pytorch实现）
- **评估指标**：准确率
- **参考资料**：[Hierarchical Attention Networks for Document Classification](https://www.cs.cmu.edu/~hovy/papers/16HLT-hierarchical-attention-networks.pdf)

#### 5. 基于Transformer的机器翻译系统
- **任务**：利用Transformer，将输入的英文句翻译成中文
- **数据集**：
	- 输入：一句英文(e.g. tom is a student.)
	- 输出：中文翻译(e.g. 汤姆 是 个 学生。)
	- 训练集：18000句
	- 验证集：500句
	- 测试集：2636句
	- 链接 #TODO 
- **要求**：Tensorflow或者 Pytorch实现（最好为Pytorch实现）
- **评价指标**：BLEU score
	- BiLingual Evaluation Understudy, IBM
	- 将机器翻译产生的候选译文与人翻译的多个参考译文相比较，越接近，候选译文的正确率越高。
	- 统计同时出现在系统译文和参考译文中的n元词的个数，最后把匹配到的n元词的数目除以系统译文的n元词数目，得到评测结果。
$$
B L E U=B P \times \exp \left(\sum_{n=1}^{N} \mathcal{w}_{n} \log p_{n}\right)
$$
	- 其中$B P$为长度过短句子的惩罚因子；$N$为最大语法的阶数，实际取4；$\mathcal{w}_n=1/N$；$p_n$表示出现在答案译文中的$n$元词语接续组占候选译文中$n$元词语接续组总数的比例。
	- $$\mathrm{BP}= \begin{cases}1 & \text { if } c>r \\ e^{(1-r / c)} & \text { if } c \leq r\end{cases}$$
	- 其中$c$为候选译文中单词的个数，$r$为答案译文中与$c$最接近的译文单词个数。
	- BLEU分值范围：0~1，分值越高表示译文质量越好，分值越小，译文质量越差。
- **参考资料**：
	- 《神经网络与深度学习》第15.4.2小节
	- [The Annotated Transformer by hardvardnlp](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
	- Vaswani, A.,Shazeer , N., Parmar, N., Uszkoreit , J., Jones, L., Gomez, A.N., Kaiser, Ł.and Polosukhin , I.. [Attention is all you](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf).In NIPS 2017.
	- Wang, Q., Li, B., Xiao, T., Zhu, J., Li, C., Wong, D. F., & Chao, L. S. (2019, July).[Learning Deep Transformer Models for Machine Translation](https://arxiv.org/pdf/1906.01787.pdf). In ACL 2019.


#### 6.基于ViT或者Swin Transformer的图像目标类别分类
- **任务**：利用ViT或者Swin Transformer，识别图像中的目标类别。
- **数据集**：[PASCAL VOC 2007挑战赛](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/)
	- 4个大类别， 20个小类别(训练验证集5011，测试集4952，共计9963幅)
		- 人类；
		- 动物（鸟、猫、牛、狗、马、羊）；
		- 交通工具（飞机、自行车、船、公共汽车、小轿车、摩托车、火车）；
		- 室内（瓶子、椅子、餐桌、盆栽植物、沙发、电视）；
	- 下载链接：
		- https://pan.baidu.com/share/init?surl=TdoXJP99RPspJrmJnSjlYg 
		- 提取码：jz27
		- 只需要下载VOC 2007的训练、验证和测试集
- **评价指标**：mAP(mean average precision)
- **要求**：Tensorflow或者 Pytorch实现（最好为Pytorch实现），可以使用最小的预训练模型
- **参考资料**：
	- Dosovitskiy, Alexey, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani et al. "[An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.](https://arxiv.org/pdf/2010.11929.pdf)" In ICLR, 2020.
	- Liu, Ze, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. "[Swin transformer: Hierarchical vision transformer using shifted windows.](https://arxiv.org/pdf/2103.14030.pdf)" In ICCV, pp. 10012-10022. 2021.
	- [Vision Transformer - Pytorch github](https://github.com/lucidrains/vit-pytorch)
	- [Swin Transformer github](https://github.com/microsoft/Swin-Transformer)
	- [数据集简介](https://blog.csdn.net/mzpmzk/article/details/88065416)

#### 7.基于预训练模型BERT的机器阅读理解任务
- **任务**：利用BERT预训练模型，实现对SQuAD v1.1的阅读理解问题。
- **数据集**：SQuAD v1.1 [下载链接](https://data.deepai.org/squad1.1.zip)
- **要求**：Tensorflow或者 Pytorch实现（最好为Pytorch实现）
- **评估指标**：Exact match(EM)、F1
- **参考**：
	- BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding [Devlin et al., NAACL 2019](https://aclanthology.org/N19-1423) 
	- 数据集来源：SQuAD: 100,000+ Questions for Machine Comprehension of Text [Rajpurkar et al., EMNLP 2016](https://aclanthology.org/D16-1264) 
