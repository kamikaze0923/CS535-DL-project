import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import matplotlib.pyplot as plt




class Sequence(nn.Module):
    def __init__(self):
        super(Sequence, self).__init__()
        self.lstm1 = nn.LSTMCell(1, 50)
        self.lstm2 = nn.LSTMCell(50, 50)
        self.linear = nn.Linear(50, 1)

    def forward(self, input, future = 0):
        outputs = []
        h_t = Variable(torch.zeros(input.size(0), 50).double(), requires_grad=False)
        c_t = Variable(torch.zeros(input.size(0), 50).double(), requires_grad=False)
        h_t2 = Variable(torch.zeros(input.size(0), 50).double(), requires_grad=False)
        c_t2 = Variable(torch.zeros(input.size(0), 50).double(), requires_grad=False)

        for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        for i in range(future):# if we should predict the future
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs




data = np.genfromtxt('./data/data.csv',delimiter=',',dtype='float32')
data = data.astype('float64')
mean = np.mean(data,axis=0)
std = np.std(data,axis=0)
data_nor = (data - mean)/std
data_nor = data_nor.transpose()


input = Variable(torch.from_numpy(data_nor[5:, :-1]), requires_grad=False)
target = Variable(torch.from_numpy(data_nor[5:, 1:]), requires_grad=False)
test_input = Variable(torch.from_numpy(data_nor[:5, :-1]), requires_grad=False)
test_target = Variable(torch.from_numpy(data_nor[:5, 1:]), requires_grad=False)



# build the model
seq = Sequence()
seq.double()
criterion = nn.MSELoss()

# use LBFGS as optimizer since we can load the whole data to train
optimizer = optim.LBFGS(seq.parameters(), lr=0.8)
#begin to train
for i in range(15):
    print('STEP: ', i)
    def closure():
        optimizer.zero_grad()
        out = seq(input)
        loss = criterion(out, target)
        print('loss:', loss.data[0])
        loss.backward()
        return loss
    optimizer.step(closure)
    # begin to predict
    future = 1000
    pred = seq(test_input, future = future)
    loss = criterion(pred[:, :-future], test_target)
    print('test loss:', loss.data)
    y = pred.data.cpu().numpy()


    def draw(yi, color):
        plt.plot(np.arange(input.size(1)), yi[:input.size(1)], color, linewidth=2.0)
        plt.plot(np.arange(input.size(1), input.size(1) + future), yi[input.size(1):], color + ':', linewidth=2.0)
    # draw the result
    plt.figure(figsize=(30, 10))
    plt.title('Predict future values for time sequences\n(Dashlines are predicted values)', fontsize=30)
    plt.xlabel('x', fontsize=20)
    plt.ylabel('y', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    draw(y[0], 'r')
    draw(y[1], 'g')
    draw(y[2], 'b')
    plt.show()
