from __future__ import print_function
from __future__ import division
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim





class Sequence(nn.Module):
    def __init__(self):
        super(Sequence, self).__init__()
        self.lstm1 = nn.LSTMCell(1, 100)
        self.lstm2 = nn.LSTMCell(100, 100)
        self.linear = nn.Linear(100, 1)

    def forward(self, input, future = 0):
        outputs = []
        h_t = Variable(torch.zeros(input.size(0), 100), requires_grad=False).cuda()
        c_t = Variable(torch.zeros(input.size(0), 100), requires_grad=False).cuda()
        h_t2 = Variable(torch.zeros(input.size(0), 100), requires_grad=False).cuda()
        c_t2 = Variable(torch.zeros(input.size(0), 100), requires_grad=False).cuda()

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
mean = np.mean(data,axis=0)
std = np.std(data,axis=0)
data_nor = (data - mean)/std
data_nor = data_nor.transpose()
data_nor = data_nor[:,600:]

future = 50
input = Variable(torch.from_numpy(data_nor[250:, :-future]), requires_grad=False).cuda()
target = Variable(torch.from_numpy(data_nor[250:, 1:-future + 1]), requires_grad=False).cuda()
test_input = Variable(torch.from_numpy(data_nor[:250, :-future]), requires_grad=False).cuda()
test_target = Variable(torch.from_numpy(data_nor[:250, 1:-future + 1]), requires_grad=False).cuda()



# build the model
seq = Sequence().cuda()
criterion = nn.MSELoss()

pretrained_dict = torch.load('mytraining.pth')
model_dict = seq.state_dict()
for i in model_dict:
    if i in pretrained_dict:
        model_dict[i] = pretrained_dict[i]
seq.load_state_dict(model_dict)


# use LBFGS as optimizer since we can load the whole data to train
optimizer = optim.LBFGS(seq.parameters(), lr=0.1)


#begin to train
for i in range(1):
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
    pred = seq(test_input, future = future)
    loss = criterion(pred[:, :-future], test_target)
    print('test loss:', loss.data)
    y = pred.data.cpu().numpy()



torch.save(seq.state_dict(), 'mytraining.pth')
for i in range(future):
    input = Variable(torch.from_numpy(data_nor[:250, :-future+i]), requires_grad=False).cuda()
    pred = seq(input, future = 1)
    y = pred.data.cpu().numpy()
    l = std.size
    if i == 0:
        pre = y[:,-1:] * std[:250].reshape(250,1) + mean[:250].reshape(250,1)
    else:
        pre = np.concatenate((pre,y[:,-1:]* std[:250].reshape(250,1) + mean[:250].reshape(250,1)),axis=1)

np.savetxt("result_ycx.csv", pre, delimiter=",")

