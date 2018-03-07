
import matplotlib.pyplot as plt
import numpy as np
future = 50

pre = np.genfromtxt('./result_ycx.csv',delimiter=',',dtype='float32')
pre = pre[:,-future:]


data = np.genfromtxt('./data/data.csv',delimiter=',',dtype='float32',usecols=tuple([i for i in range(250)]))
act = data.transpose()
act = act[:,-future:]


def draw(pre, act, color, label1, label2):
    plt.plot(np.arange(future), act, color=color, linewidth=1, label=label1)
    plt.plot(np.arange(future), pre, color=color, linestyle=':', linewidth=1, label=label2)
    plt.scatter(np.arange(future), act, marker='+')
    plt.scatter(np.arange(future), pre, marker='*')


# draw the result
plt.figure(figsize=(30, 10))
plt.title('Predict future values for time sequences\n(Dashlines are predicted values)', fontsize=30)
plt.xlabel('x', fontsize=20)
plt.ylabel('y', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
draw(pre[6], act[6], 'blue', 'act_open', 'pre_open')
# draw(pre[1], act[1], 'orange', 'act_close', 'pre_close')
# draw(pre[2], act[2], 'green', 'act_low', 'pre_low')
# draw(pre[3], act[3], 'red', 'act_high', 'pre_high')
# draw(pre[4], act[4], 'purple', 'volume')
legend = plt.legend(loc='0')
plt.grid(linewidth=1)
plt.show()