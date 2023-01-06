import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import numpy as np
depth = (7,8,9,10)
x = np.linspace(7,10,20)
hash_1 = (5.308, 6.332, 9.426, 16)
hash_2 = (4.733, 6.275, 8.984, 15)
ori_1 = (5.315, 7.17, 10, 16.4)
ori_2 = (5.008, 6.696, 9.2, 15.4)
fig = plt.figure()
fig.suptitle('Time consumption comparison')
plt.plot(depth,hash_1,label="hash-strategy1",c = "black")
plt.plot(depth,hash_2,label="hash-strategy2", c = "orange")
plt.plot(depth,ori_1,label="strategy1", c = "blue")
plt.plot(depth,ori_2,label="strategy2", c = "c")
plt.plot(x, [6.332 for i in x], linestyle='--', c='red')
plt.ylabel("Time per step(s)")
plt.xlabel("depth(layers)")
plt.legend()
x = MultipleLocator(1) 
ax=plt.gca()
#ax为两条坐标轴的实例
ax.xaxis.set_major_locator(x)
#把x轴的主刻度设置为1的倍数
plt.show()
fig.savefig("plot.png")