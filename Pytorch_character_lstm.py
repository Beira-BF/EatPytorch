# 一、Pytorch中的LSTM
# 使用Pytorch中自带的lstm模型时需要注意的点：
# 1、Pytorch中自带的LSTM的输入形式是一个3D的Tensor，每一个纬度都有重要的意义，第一个纬度就是序列本身，第二个纬度是
# mini-batch中实例的索引，第三个纬度是输入元素的索引
# 2、embedding层的输出一般是一个二维的向量，因此如果embedding层后直接接lstm，需要利用tensor.view()方法显式的修改
# embedding层的输出结构——embeds.view(len(sentence), 1, -1)
# 3、LSTM的输出格式为output, (h_n, c_n),其中output保存了最后一层，每个time step的输出h，如果是双向LSTM，每个time
# step的输出h=[h正向，h逆向](同一个time step的正向和逆向的h连接起来);而h_n实际上是每个layer最后一个状态（纵向）输出的
# 拼接，c_n为每个layer最后一个状态记忆单元中的值的拼接。
# 详细链接：https://zhuanlan.zhihu.com/p/79064602

# 二、用LSTM来进行词性标注
# Pytorch教程中给出了LSTM网络进行词性标注的代码，代码包括了三部分——数据准备，创建模型于训练模型，
# 2.1 数据准备
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

training_data = [
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
]
word_to_ix = {}
for sent, tags in training_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
print(word_to_ix)
tag_to_ix = {"DET":0, "NN": 1, "V": 2}

# 实际中通常使用更大的纬度如32维，64维，
# 这里我们使用小的纬度，为了方便查看训练过程中权重的变化。
EMBEDDING_DIM = 6
HIDDEN_DIM = 6

# 2.2 创建模型

class LSTMTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        # LSTM以word_embeddings作为输入，输出纬度为hidden_dim的隐藏状态值
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        # 线性层将隐藏状态空间映射到标注空间
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # 一开始并没有隐藏状态所以我们要先初始化一个
        # 关于纬度为什么这么设计请参考Pytorch相关文档
        # 各个维度的含义是（num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm(embeds.view(len(sentence), 1, -1), self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

# 2.3 模型训练
model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)


# 查看训练前的分数
# 注意：输出的i，j元素的值表示单词i的j标签的得分
# 这里我们不需要训练，不需要求导，所以使用torch.no_grad()
with torch.no_grad():
    inputs = prepare_sequence(training_data[0][0], word_to_ix)
    tag_scores = model(inputs)
    print(tag_scores)

for epoch in range(300):     # 实际情况下不会训练300个周期，此例中我们随便设一个值
    for sentence, tags in training_data:
        # 第一步：请记住Pytorch会累加梯度
        # 我们需要在训练每个实例前清空梯度
        model.zero_grad()

        # 此外还需要清空LSTM的隐状态，
        # 将其从上个实例的历史中分离出来。
        model.hidden = model.init_hidden()

        # 准备网络的输入，将其变为词索引的Tensor类型数据
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = prepare_sequence(tags, tag_to_ix)

        # 第三步：前向传播
        tag_scores = model(sentence_in)

        # 第四步：计算损失和梯度值，通过调用optimizer.step()来更新梯度
        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()

# 查看训练后的得分
with torch.no_grad():
    inputs = prepare_sequence(training_data[0][0], word_to_ix)
    tag_scores = model(inputs)

    # 句子是"the dog ate the apple", i,j表示对于单词i，标签j的得分，
    # 我们采用得分最高的标签作为预测的标签，从下面的输出我们可以看到，预测
    # 得到的结果是0，1，2，0，1，因为索引是从0开始的，因此第一个值0表示第一行的
    # 最大值，第二个值1表示第二行的最大值，以此类推，所以最后的结果是:
    # DET NOUN VERB DET NOUN, 整个序列都是正确的
    print(tag_scores)

# 三、使用字符级特征来增强LSTM次性标注器
# 教程给出的模型只使用词向量作为序列模型的输入，相当于只考虑词级别的特征，而像词缀这样的字符级信息对于次性有很大的影响。
# 比如，包含词缀-ly的单词基本上都是标注为副词。因此，接下来我们会在刚刚代码的基础上考虑加入每个单词的字符级别的特征来增强词嵌入。
# 教程中给出的思路为：
# 新模型中需要两个LSTM，一个跟之前一样，用来输入词性标注的得分，另外一个新增加的用来获取每个单词的字符级别表达；
# 为了在字符级别上运行序列模型，需要用嵌入的字符来作为字符LSTM的输入。
# 因此，我们会在模型设置两个embedding层——character_embeddings与word_embedding:

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

training_data = [
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
]
word_to_ix = {}
character_to_ix = {}
words_characters = {}

for sent, tags in training_data:
    for word in sent:
        if word not in word_to_ix:
            words_characters[len(word_to_ix)] = list(word)
            word_to_ix[word] = len(word_to_ix)

            for character in word:
                if character not in character_to_ix:
                    character_to_ix[character] = len(character_to_ix)

print(character_to_ix)
tag_to_ix = {"DET":0, "NN": 1, "V": 2}

# 实际中通常使用更大的纬度如32维，64维，
# 这里我们使用小的纬度，为了方便查看训练过程中权重的变化。
WORD_EMBEDDING_DIM = 6
CHARACTER_EMBEDDING_DIM = 3
HIDDEN_DIM = 6
CHARACTER_HIDDEN_DIM = 3

class LSTMTaggerCharacter(nn.Module):
    def __init__(self, embedding_dim, character_embedding_dim, hidden_dim, character_hidden_dim, vocab_size,
                 character_size, tagset_size):
        super(LSTMTaggerCharacter, self).__init__()
        self.hidden_dim = hidden_dim
        self.character_hidden_dim = character_hidden_dim

        # 词嵌入
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # 字符嵌入
        self.character_embeddings = nn.Embedding(character_size, character_embedding_dim)

        # lstm_character以每个字符的character_embedding作为输入，输出即为该单词对应字符级别的特征，
        # 输出纬度为character_hidden_dim的隐藏状态值
        self.lstm_character = nn.LSTM(character_embedding_dim, character_hidden_dim)

        # tag_lstm以word_embeddings和该词字符级别特征的拼接向量作为输入，输出纬度为hidden_dim的隐藏状态值
        self.tag_lstm = nn.LSTM(embedding_dim + character_hidden_dim, hidden_dim)

        # 线性层将隐藏状态空间映射到标注空间
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden_tag = self.init_hidden(hidden_dim)
        self.hidden_character = self.init_hidden(character_hidden_dim)

    def init_hidden(self, hidden_dim):
        # 一开始并没有隐藏状态，所以我们需要先初始化一个
        # 关于纬度为什么这么设计，请参考Pytorch相关文档
        # 各个维度的含义是(num_layers, minibatch_size, hidden_dim）
        return (torch.zeros(1, 1, hidden_dim),
                torch.zeros(1, 1, hidden_dim))

    def forward(self, sentence, words_characters):
        embeds = list()
        for sentenve_word in sentence:
            # 词嵌入
            word_embed = self.word_embeddings(sentenve_word)

            # 获取单词字符级别的特征
            word_character = words_characters[sentenve_word.item()]
            word_character_in = prepare_sequence(word_character, character_to_ix)
            character_embeds = self.character_embeddings(word_character_in)
            character_lstm_out, self.hidden_character = self.lstm_character(
                character_embeds.view(len(word_character_in), 1, -1), self.hidden_character)

            # 拼接词向量与字符级别的特征
            embed = torch.cat((word_embed, self.hidden_character[0].view(-1)))
            embeds.append(embed)

        # 拼接句子中每个词的词向量，拼接后的结果作为tag_lstm的输入
        embeds = torch.cat(embeds).view(len(sentence), 1, -1)
        lstm_out, self.hidden_tag = self.tag_lstm(embeds, self.hidden_tag)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

model = LSTMTaggerCharacter(WORD_EMBEDDING_DIM, CHARACTER_EMBEDDING_DIM, HIDDEN_DIM, CHARACTER_HIDDEN_DIM,
                            len(word_to_ix), len(character_to_ix), len(tag_to_ix))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 查看训练前的分数
# 注意：输出的i，j元素的值表示单词i的j标签的得分
# 这里我们不需要训练不需要求导，所以使用torch.no_grad()
with torch.no_grad():
    inputs = prepare_sequence(training_data[0][0], word_to_ix)
    tag_scores = model(inputs, words_characters)
    print("before training: \n")
    print(tag_scores)

for epoch in range(300):
    for sentence, tags in training_data:
        model.zero_grad()

        model.hidden_tag = model.init_hidden(HIDDEN_DIM)
        model.hidden_character = model.init_hidden(CHARACTER_HIDDEN_DIM)

        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = prepare_sequence(tags, tag_to_ix)

        tag_scores = model(sentence_in, words_characters)

        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()

with torch.no_grad():
    inputs = prepare_sequence(training_data[0][0], word_to_ix)
    tag_scores = model(inputs, words_characters)
    print("after training: \n")
    print(tag_scores)