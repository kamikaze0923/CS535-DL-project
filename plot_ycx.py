
import matplotlib.pyplot as plt
import numpy as np
future = 300

pre = np.genfromtxt('./result_ycx.csv',delimiter=',',dtype='float32')
pre = pre[:,-future:]



data = np.genfromtxt('./data/data.csv',delimiter=',',dtype='float32',usecols=[i*5+1 for i in range(483)])
act = data.transpose()
act = act[:,-future:]




def draw(pre, act, label1, label2):
    plt.plot(np.arange(future), act, color='r', linewidth=1, label=label1)
    plt.scatter(np.arange(future), act, marker='+')
    plt.plot(np.arange(future), pre, color='b', linestyle=':', linewidth=1, label=label2)
    plt.scatter(np.arange(future), pre, marker='*')

n = 1
# draw the result
plt.figure(figsize=(30, 10))
plt.title('Stock %s Last 300 days prediction\n' % str(n+1), fontsize=30)
plt.xlabel('x', fontsize=20)
plt.ylabel('y', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)



draw(pre[n], act[n], 'Actual', 'Predicted')

legend = plt.legend(loc='0')
plt.grid(linewidth=1)
plt.show()