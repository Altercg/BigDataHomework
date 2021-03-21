import torch

# 构造5x3的矩阵，不初始化
result = torch.empty(3,3)    
# 构造一个随机初始化的矩阵
x = torch.rand(5,3)   
# 直接用数据构造一个张量
x = torch.tensor([5.5,3])   
# 数据类型是int32的0矩阵
y = torch.zeros(3,3,dtype=torch.float)   
# 默认返回一个与y有同样torch.dtype和torch.device的size为4x3的，值为1的tensor，记得要赋值
x = y.new_ones(3,3)
# 与y一样尺寸，创建随机值的tensor
y = torch.randn_like(y, dtype=torch.float)  
# 获取x的维度信息
print(x.size())

# tensor可以进行加法
# 直接加
x+y
# 存放在存在的result中，大小需要一致
torch.add(x,y, out=result)
# 存放在y中
y.add_(x)
# 索引操作
print(y[:,1])
# 改变tensor的大小或者形状
x = torch.randn(6, 4)
y = x.view(24)
z = x.view(-1, 8)  # -1意味着它的维度随着别的维度改变，输出结果为[3,8]
# 包含一个元素的tensor使用.item()来获得其value
x = torch.randn(1)
print(x.item())
# 自动微分，参考：https://www.cnblogs.com/cocode/p/10746347.html
#创建一个张量，设置requires_grad=True来跟踪与它相关的计算
x = torch.ones(2, 2, requires_grad=True)
y = x + 2   # grad_fn=<AddBackward0> 加操作得
z = y*y*3
out = z.mean()
# 反向传播，应用链式法则计算梯度
out.backward()
# 打印梯度 d(out)/dx
print(x.grad)
#将代码包裹在with torch.no_grad()，来停止对从跟踪历史中的 .requires_grad=True的张量自动求导。
with torch.no_grad():
    print((x ** 2).requires_grad)
# 训练神经网络
# 一个模型可训练的参数可以通过调用 net.parameters()
params = list(net.parameters())
print(len(params))
print(params[0].size())  # conv1's .weight
# 把所有参数梯度缓存器置零，用随机的梯度来反向传播
net.zero_grad()
out.backward(torch.randn(1, 10))

# 返回数字个数
a = torch.randn(1, 2, 3, 4, 5)  # 4行5列，3个一组，有两组，两组是一个tensor
print(torch.numel(a))
# numpy转为tensor，t变了，a也会变
a = numpy.array([1, 2, 3])
t = torch.from_numpy(a)
# 创建[0,5)间隔为2
torch.arange(0, 5, 2)   
# 创建[1,4]间隔为0.5
torch.range(1, 4,0.5)
# 一维[3,10]，5个，等分
torch.linspace(3, 10, steps=5)
# 3*3的E
torch.eye(3)
# 按值初始化
torch.full((2, 3), 3.141592)