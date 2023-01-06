import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
depth = (2,3,4)
x_ori = (154,2.64*1000, 14*1000)
x_ng = (94, 614, 2616)
x_g = (88, 300, 1686)
x_z = (83, 200, 1022)
fig = plt.figure()
fig.suptitle('Time per step')
plt.plot(depth,x_ori,label="Original",c = "black")
plt.plot(depth,x_ng,label="Sorted", c = "orange")
plt.plot(depth,x_g,label="Greedy", c = "blue")
plt.plot(depth,x_z,label="Zobrist", c = "red")
plt.ylabel("Time per step(ms)")
plt.xlabel("depth(layers)")
plt.legend()
x = MultipleLocator(1) 
ax=plt.gca()
#ax为两条坐标轴的实例
ax.xaxis.set_major_locator(x)
#把x轴的主刻度设置为1的倍数
plt.show()
fig.savefig("plot.png")