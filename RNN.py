'''
训练RNN模型使得  "hello" -> "ohlol"
输入为"hello"，可设置字典 e -> 0 h -> 1 l -> 2 o -> 3 hello对应为 10223 one-hot编码有下面对应关系
h   1   0100            o   3
e   0   1000            h   1
l   2   0010            l   2
l   2   0010            o   3
o   3   0001            l   2
输入有“helo”四个不同特征于是input_size = 4
hidden_size = 4 batch_size = 1

RNN模型维度的确认至关重要：
rnn = torch.nn.RNN(input_size=input_size, hidden_size=hidden_size,num_layers=num_layers)
outputs, hidden_outs = rnn(inputs, hiddens):
    inputs of shape 𝑠𝑒𝑞𝑆𝑖𝑧𝑒, 𝑏𝑎𝑡𝑐ℎ, 𝑖𝑛𝑝𝑢𝑡_𝑠𝑖𝑧𝑒
    hiddens of shape 𝑛𝑢𝑚𝐿𝑎𝑦𝑒𝑟𝑠, 𝑏𝑎𝑡𝑐ℎ, ℎ𝑖𝑑𝑑𝑒𝑛_𝑠𝑖𝑧𝑒
    outputs of shape 𝑠𝑒𝑞𝑆𝑖𝑧𝑒, 𝑏𝑎𝑡𝑐ℎ, ℎ𝑖𝑑𝑑𝑒𝑛_𝑠𝑖𝑧𝑒
    hidden_outs of shape 𝑠𝑒𝑞𝑆𝑖𝑧𝑒, 𝑏𝑎𝑡𝑐ℎ, ℎ𝑖𝑑𝑑𝑒𝑛_𝑠𝑖𝑧𝑒
cell = torch.nn.RNNcell(input_size=input_size, hidden_size=hidden_size)
output, hidden_out = cell(input, hidden):
    input of shape 𝑏𝑎𝑡𝑐ℎ, 𝑖𝑛𝑝𝑢𝑡_𝑠𝑖𝑧𝑒
    hidden of shape 𝑏𝑎𝑡𝑐ℎ, ℎ𝑖𝑑𝑑𝑒𝑛_𝑠𝑖𝑧𝑒
    output of shape 𝑏𝑎𝑡𝑐ℎ, ℎ𝑖𝑑𝑑𝑒𝑛_𝑠𝑖𝑧𝑒
    hidden_out of shape 𝑏𝑎𝑡𝑐ℎ, ℎ𝑖𝑑𝑑𝑒𝑛_𝑠𝑖𝑧𝑒
其中，seqSize：输入个数  batch：批量大小  input_size：特征维数 numLayers：网络层数  hidden_size：隐藏层维数
'''
import torch

idx2char = ['e', 'h', 'l', 'o']  # 方便最后输出结果
x_data = [1, 0, 2, 2, 3]  # 输入向量
y_data = [3, 1, 2, 3, 2]  # 标签

one_hot_lookup = [[1, 0, 0, 0],  # 查询ont hot编码 方便转换
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]]
x_one_hot = [one_hot_lookup[x] for x in x_data]  # 按"1 0 2 2 3"顺序取one_hot_lookup中的值赋给x_one_hot
'''运行结果为x_one_hot = [ [0, 1, 0, 0],
                          [1, 0, 0, 0],
                          [0, 0, 1, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1] ]
刚好对应输入向量，也对应着字符值'hello'
'''


def cell():
    input_size = 4
    hidden_size = 4
    batch_size = 1
    inputs = torch.Tensor(x_one_hot).view(-1, batch_size, input_size)
    labels = torch.LongTensor(y_data).view(-1, 1)  # 增加维度方便计算loss

    class cell_Model(torch.nn.Module):
        def __init__(self, input_size, hidden_size, batch_size):
            super(cell_Model, self).__init__()
            self.input_size = input_size
            self.batch_size = batch_size
            self.hidden_size = hidden_size
            self.rnncell = torch.nn.RNNCell(input_size=self.input_size, hidden_size=self.hidden_size)

        def forward(self, input, hidden):
            hidden = self.rnncell(input, hidden)  # shape: 𝑏𝑎𝑡𝑐ℎ, 𝑖𝑛𝑝𝑢𝑡_𝑠𝑖𝑧𝑒
            return hidden

        def init_hidden(self):
            return torch.zeros(self.batch_size, self.hidden_size)  # 提供初始化隐藏层（h0）

    net = cell_Model(input_size, hidden_size, batch_size)

    # ---计算损失和更新
    criterion = torch.nn.CrossEntropyLoss()  # 交叉熵
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    # ---计算损失和更新

    for epoch in range(50):  # 训练50次
        loss = 0
        optimizer.zero_grad()
        hidden = net.init_hidden()
        print('Predicten string:', end='')
        for input, label in zip(inputs, labels):  # 并行遍历数据集 一个一个训练
            hidden = net(input, hidden)  # shape: 𝑏𝑎𝑡𝑐ℎ, 𝑖𝑛𝑝𝑢𝑡_𝑠𝑖𝑧𝑒        𝑏𝑎𝑡𝑐ℎ, ℎ𝑖𝑑𝑑𝑒𝑛_𝑠𝑖𝑧𝑒
            # hidden输出维度 𝑏𝑎𝑡𝑐ℎ, ℎ𝑖𝑑𝑑𝑒𝑛_𝑠𝑖𝑧𝑒
            loss += criterion(hidden, label)
            _, idx = hidden.max(dim=1)  # 从第一个维度上取出预测概率最大的值和该值所在序号
            print(idx2char[idx.item()], end='')  # 按上面序号输出相应字母字符
        loss.backward()
        optimizer.step()
        print(', Epoch [%d/50] loss=%.4f' % (epoch + 1, loss.item()))


def RNN_module():
    input_size = 4
    hidden_size = 4
    num_layers = 1
    batch_size = 1
    seq_len = 5
    inputs = torch.Tensor(x_one_hot).view(seq_len, batch_size, input_size)
    labels = torch.LongTensor(y_data)

    class RNNModel(torch.nn.Module):
        def __init__(self, input_size, hidden_size, batch_size, num_layers=1):
            super(RNNModel, self).__init__()
            self.num_layers = num_layers
            self.input_size = input_size
            self.batch_size = batch_size
            self.hidden_size = hidden_size
            self.rnn = torch.nn.RNN(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=num_layers)

        def forward(self, input):
            hidden = torch.zeros(self.num_layers, self.batch_size, self.hidden_size)  # 提供初始化隐藏层（h0）
            out, _ = self.rnn(input, hidden)  # out=[ h0, h1, h2, h3, h4]  _ = [[[h4]]]
            return out.view(-1, self.hidden_size)

    net = RNNModel(input_size, hidden_size, batch_size, num_layers)
    # ---计算损失和更新
    criterion = torch.nn.CrossEntropyLoss()  # 交叉熵
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    # ---计算损失和更新

    for epoch in range(100):  # 训练100次
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, idx = outputs.max(dim=1)  ##从第一个维度上取出预测概率最大的值和该值所在序号
        idx = idx.data.numpy()
        print('Predicted: ', ''.join([idx2char[x] for x in idx]), end='')
        print(', Epoch [%d/100] loss = %.3f' % (epoch + 1, loss.item()))


def embedding():
    num_class = 4  # 类别数量
    input_size = 4  # 输入维度
    hidden_size = 8  # 隐藏层维度
    embedding_size = 10  # 嵌入到10维空间
    num_layers = 2  # RNN层数
    batch_size = 1
    seq_len = 5  # 数据量
    idx2char = ['e', 'h', 'l', 'o']  # 方便最后输出结果
    x_data = [[1, 0, 2, 2, 3]]  # (batch, seq_len)
    y_data = [3, 1, 2, 3, 2]  # (batch * seq_len)

    inputs = torch.LongTensor(x_data)
    labels = torch.LongTensor(y_data)

    class Model(torch.nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.emb = torch.nn.Embedding(input_size, embedding_size)
            self.rnn = torch.nn.RNN(input_size=embedding_size,
                                    hidden_size=hidden_size,
                                    num_layers=num_layers,
                                    batch_first=True)
            self.fc = torch.nn.Linear(hidden_size, num_class)

        def forward(self, x):
            hidden = torch.zeros(num_layers, x.size(0), hidden_size)
            x = self.emb(x)  # input (batch,seqLen)  output (batch, seqLen, embeddingSize)
            x, _ = self.rnn(x, hidden)
            x = self.fc(x)
            return x.view(-1, num_class)  # 修改维度好用交叉熵计算损失

    net = Model()
    # ---计算损失和更新
    criterion = torch.nn.CrossEntropyLoss()  # 交叉熵
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    # ---计算损失和更新

    for epoch in range(15):  # 训练15次
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, idx = outputs.max(dim=1)  ##从第一个维度上取出预测概率最大的值和该值所在序号
        idx = idx.data.numpy()
        print('Predicted: ', ''.join([idx2char[x] for x in idx]), end='')
        print(', Epoch [%d/15] loss = %.3f' % (epoch + 1, loss.item()))


if __name__ == '__main__':
    # cell()
    # RNN_module()
    embedding()
