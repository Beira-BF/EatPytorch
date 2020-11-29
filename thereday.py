# 1-3, 文本数据建模流程范例

# 一、准备数据
import torch
import string, re
import torchtext

MAX_WORDS = 10000  # 仅考虑最高频的10000个词
MAX_LEN = 200 # 每个样本保留200个词的长度
BATCH_SIZE = 20

# 分词方法
tokenizer = lambda x:re.sub('[%s]'%string.punctuation,"",x).split(" ")

# 过滤掉低频词
def filterLowFreqWords(arr, vocab):
    arr = [[x if x<MAX_WORDS else 0 for x in example] for example in arr]
    return arr

# 1, 定义各个字段的预处理方法
TEXT = torchtext.data.Field(sequential=True, tokenize=tokenizer, lower=True,
                            fix_length = MAX_LEN, posprecessing=filterLowFreqWords)

LABEL = torchtext.data.Field(sequential=False, use_vocab=False)

# 2, 构建表格型dataset
# torchtext.data.TabularDataset可读取csv, tsv, json等格式
ds_train, ds_test = torchtext.data.TabularDataset.splits(
    path='./data/imdb', train='train.tsv', test='test.tsv', format='tsv',
    fields=[('label', LABEL), ('text', TEXT)], skip_header=False)

# 3. 构建字典
TEXT.build_vocab(ds_train)

# 4. 构建数据管道迭代器
train_iter, test_iter = torchtext.data.Iterator.splits(
    (ds_train, ds_test), sort_within_batch=True, sort_key=lambda x: len(x.text),
    batch_sizes=(BATCH_SIZE,BATCH_SIZE))

# 查看example信息
print(ds_train[0].text)
print(s_train[0].label)

# 查看字典信息
print(len(TEXT.vocab))

# itos: index to string
print(TEXT.vocab.itos[0])
print(TEXT.vocab.itos[1])

# stoi: string to index
print(TEXT.vocab.stoi['<unk>'])  # unknown 未知词
print(TEXT.vocab.stoi['<pad>'])  # padding 填充

# freqs: 词频
print(TEXT.vocab.freqs['<unk>'])
print(TEXT.vocab.freqs['a'])
print(TEXT.vocab.freqs['good'])

# 查看数据管道信息
# 注意有坑：text第0维是句子长度
for batch in train_iter:
    features = batch.text
    labels = batch.label
    print(features)
    print(features.shape)
    print(labels)
    break

class DataLoader:
    def __init__(self, data_iter):
        self.data_iter = data_iter
        self.length = len(data_iter)

    def __len__(self):
        return self.length

    def __iter__(self):
        # 注意：此处调整features为batch_first, 并调整label的shape和dtype
        for batch in self.data_iter:
            yield (torch.transpose(batch.text, 0, 1),
                   torch.unsqueeze(Batch.label.float(), dim = 1))

dl_train = DataLoader(train_iter)
dl_test = DataLoader(test_iter)

# 定义模型
# 使用Pytorch通常由三种方式构建模型：使用nn.Sequential按层顺序构建模型，继承nn.Module基类构建自定义模型，继承nn.Module基类构建
# 模型并辅助应用模型容器（nn.Sequential, nn.ModuleList, nn.ModuleDict)进行封装。
# 此处选择使用第三种方式进行构建。
# 由于接下来使用类形式的训练循环，我们将模封装成torchkeras.Model类来获得类似Kera中高阶模型接口的功能。
# Model类实际上继承自nn.Module类
import torch
from torch import nn
import torchkeras

torch.random.seed()

class Net(torchkeras.Model):
    def __init__(self):
        super(Net, self).__init__()

        # 设置padding_idx参数后将在训练过程中将填充的token始终赋值为0向量
        self.embedding = nn.Embedding(num_enbeddings=MAX_WORDS, embedding_dim=3, padding_idx = 1)
        self.conv = nn.Sequential()
        self.conv.add_module("conv_1", nn.Conv1d(in_channels=3, out_chennels=16, kernal_size=5))
        self.conv.add_module("pool_1", nn.MaxPool1d(kernal_size=2))
        self.conv.add_module("relu_1", nn.ReLU())
        self.conv.add_module("conv_2", nn.Conv1d(in_channels=16, out_channels=128, kernal_size=2))
        self.conv.add_module("pool_2", nn.MaxPool1d(kernel_size=2))
        self.conv.add_module("relu_2", nn.ReLU())

        self.dense = nn.Sequential()
        self.dense.add_module("flatten", nn.Flatten())
        self.dense.add_module("linear", nn.Linear(6144, 1))
        self.dense.add_module("sigmoid", nn.Sigmoid())

    def forward(self,x):
        x = self.embedding(x).transpose(1,2)
        x = self.conv(x)
        y = self.dense(x)

        return y

model = Net()
print(model)

model.summary(input_shape=(200,), input_dtype = torch.LongTensor)


# 训练模型
# 训练Pytorch通常需要用户编写自定义训练循环，训练循环的代码风格因人而异
# 有三类典型的训练循环代码风格：脚本形式训练循环，函数形式训练循环，类形式训练循环
# 此处介绍一种类形式的训练循环。
# 我们仿照Keras定义了一个高阶的模型接口Model，实现fit，validate，predict，summary方法，相当于用户自定义高阶API

# 准确率
def accuracy(y_pred, y_true):
    y_pred = torch.where(y_pred>0.5, torch.ones_like(y_pred, dtype=torch.float32),
                         torch.zero_like(y_pred, dtype=torch.float32))
    acc = torch.mean(1-torch.abs(y_true-y_pred))
    return acc
model.compile(loss_func=nn.BCELoss(), optimizer=torch.optim.Adagrad(model.parameters(), lr=0.02),
              metrics_dict={"accuracy":accuracy})

# 有时候模型训练过程不收敛，需要多试几次
dfhistory = model.fit(20, dl_train, dl_val=dl_test, log_step_freq=200)

# 评估模型
import matplotlib.pyplot as plt

def plot_metric(dfhistory, metric):
    train_metrics = dfhistory[metric]
    val_metrics = dfhistory['val_'+metric]
    epochs = range(1,len(train_metrics)+1)
    plt.plot(epochs, train_metrics, 'bo--')
    plt.plot(epochs, val_metrics, 'ro-')
    plt.title('Training and validation '+ metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_"+metric, 'val_'+metric])
    plt.show()

plot_metric(dfhistory, "loss")
plot_metric(dfhistory, "accuracy")

# 评估
model.evaluate(dl_test)

# 使用模型
model.predict(dl_test)

# 保存模型
print(model.state_dict().keys())

# 保存模型参数
torch.save(model.state_dict(), "./data/model_parameter.pkl")

model_clone = Net()
model_clone.load_state_dict(torch.load("./data/model_parameter.pkl"))

model_clone.compile(loss_func = nn.BCELoss(), optimizer=torch.optim.Adagrad(model.parameters(), lr=0.02),
                    metric_dict={"accuracy":accuracy})

# 评估模型
model_clone.evaluate(dl_test)
