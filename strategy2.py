import numpy as np

future = 300
n = 483

ind = np.genfromtxt('./indicator_ycx1.csv',delimiter=',',dtype='float32')
ind = ind[:,-future:]
idx = np.unravel_index(np.argmin(ind, axis=None), ind.shape)
print(idx)
print(np.max(ind),np.min(ind))

data = np.genfromtxt('./data/data.csv',delimiter=',',dtype='float32',usecols=[i*5+1 for i in range(483)])
act = data.transpose()
act = act[:,-future:]

buy_threshold = np.arange(0,0.056,0.00001)
sell_threshold = np.arange(0,-0.056,-0.00001)
in_hand_stocks = {}



rate = np.empty(shape=(5600,5600))

for X,x in enumerate(np.nditer(buy_threshold)):
    for Y,y in enumerate(np.nditer(sell_threshold)):
        pay = 0
        back = 0
        for i in range(future):
            sell_stocks = []
            for k in in_hand_stocks:
                if ind[k,i] < y or i == future - 1:
                    # print('Sell stock %d in day %d at %f with buy price %f, profit_rate=%f' % (k, i, act[k,i], in_hand_stocks[k], (act[k,i] - in_hand_stocks[k])/in_hand_stocks[k]))
                    back += act[k,i]
                    sell_stocks.append(k)
            for j in sell_stocks:
                in_hand_stocks.pop(j)
            for j in range(n):
                if ind[j,i] > x and j not in in_hand_stocks and (i != future - 1):
                    # print('Buy stock %d in day %d at %f' % (j, i, act[j, i],))
                    in_hand_stocks[j] = act[j, i]
                    pay += act[j,i]

        # print('Cost: ' + str(pay))
        # print('Income: ' + str(back))
        profit = (back-pay)/pay if pay != 0 else 0
        print(('Profit Rate: %4f' % profit) + (' Buy_threshold: %3f' % x) + (' Sell_threshold: %3f' % y))

        rate[X,Y] = profit
print(rate)
np.savetxt("rate_ycx.csv", rate, delimiter=",")


