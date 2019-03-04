1、numpy where的用法：
print("------np.where--------------")
用法一：np.where(condition, x, y) 满足条件(condition)，输出x，不满足输出y。
np.where([[True,False], [True,True]],    # 官网上的例子
             [[1,2], [3,4]],
             [[9,8], [7,6]])
array([[1, 8],
       [3, 4]])
	   
用法二：np.where(condition)
a = np.array([2,4,6,8,10])
np.where(a > 5)  # (array([2, 3, 4]),)
np.where((a > 5) & (a < 7))  # 注意&的用法

返回的是元组，里面是索引，一维返回一维的，二维返回二维的
np.where([[True,False], [True,True]]

ix 为 array([[False, False, False],
       [ True,  True, False],
       [False,  True, False]], dtype=bool)
则：
np.where(ix) #(array([1, 1, 2]), array([0, 1, 1]))，得到的是变量为true的索引

总结：
a > 6            ==> [False, False, False,True,  True]
a[a>6]           ==> [8, 10]

index = np.where(a > 6)  ==> (array([3,4]),)
a[index]           ==> [8, 10]
#a>6 和 np.where(a > 6) 都是获取索引，结果形式不一样
#a[a>6]和a[index] 都可以获得一样结果
# 应用实例：a = np.ones(5) b = np.random.rand((5, 10)) b[a>0]可以获得对应的二维数据

2、np.argmax np.max np.maximum np.sort np.argsort
a = np.arange(20).reshape(5, 2, 2)
print("------np.argmax and np.max--------------")
index = np.argmax(a, axis=0)  # 求第一维度上最大值对应的索引
print(index) 
[[4 4]
 [4 4]]
print(index.shape) #(2, 2)  如果 index = np.argmax(a, axis=1)，则对应的维度是（5， 2）

max_value = np.max(a, axis=0) # 获取在第一维度中的最大值
print(max_value)
[[16 17]
 [18 19]]
 
 print("------np.sort--------------")
a = np.array([[1, 4, 2], [3, 1, 3]])
print(np.sort(a))  #等价于np.sort(a, axis=1)  默认按照最后一个维度排序， 返回值
print(np.sort(a, axis=1))
[[1 2 4]
 [1 3 3]]

print(np.sort(a, axis=None)) # flat [1 1 2 3 3 4]

print(np.sort(a, axis=0)) #纵向比价 
[[1 1 2]
 [3 4 3]]

print("------np.argsort --------------")
print(np.argsort(a)) # 默认按照最后一个维度
print(np.argsort(a, axis=1))
[[0 2 1]
 [1 0 2]]
print(np.argsort(a, axis=0))
[[0 1 0]
 [1 0 1]]
print(np.argsort(a, axis=None))  # flat
[0 4 2 3 5 1]

print("------structured array --------------")
# https://blog.csdn.net/qq_27825451/article/details/82425512
dtype = [('name', 'S10'), ('height', float), ('age', int)]
values = [('Arthur', 1.8, 41), ('Lancelot', 1.9, 38),('Galahad', 1.7, 38)]
a = np.array(values, dtype=dtype)

print(a.shape)
print(np.sort(a))  # 等价于np.sort(a, order='name')
print(np.sort(a, order='height'))

temp = np.where(a['height']>1.7)
print(temp)

a = numpy.array([1,2,3])
print(a[...] == a) # [true, true, true]  a[...]等价于a
print("------np.maximum： --------------")
np.maximum(X, Y, out=None) #X 与 Y 逐位比较取其大者；最少接收两个参数
np.maximum([-2, -1, 0, 1, 2], 0)  # array([0, 0, 0, 1, 2])
# 当然 np.maximum 接受的两个参数，也可以大小一致, 或者更为准确地说，第二个参数只是一个单独的值时，其实是用到了维度的 broadcast 机制

""""
#pytorch
torch.nn的核心数据结构是Module，它是一个抽象概念，既可以表示神经网络中的某个层（layer），也可以表示一个包含很多层的神经网络。
在实际使用中，最常见的做法是继承nn.Module，撰写自己的网络/层。
自定义层Linear必须继承nn.Module，并且在其构造函数中需调用nn.Module的构造函数，即super(Linear, self).__init__() 或nn.Module.__init__(self)，推荐使用第一种用法。
在构造函数__init__中必须自己定义可学习的参数，并封装成Parameter，如在本例中我们把w和b封装成parameter。parameter是一种特殊的Variable，但其默认需要求导（requires_grad = True）。
forward函数实现前向传播过程，其输入可以是一个或多个variable，对x的任何操作也必须是variable支持的操作。
无需写反向传播函数，因其前向传播都是对variable进行操作，nn.Module能够利用autograd自动实现反向传播，这点比Function简单许多。
使用时，直观上可将layer看成数学概念中的函数，调用layer(input)即可得到input对应的结果。它等价于layers.__call__(input)，在__call__函数中，主要调用的是 layer.forward(x)，另外还对钩子做了一些处理。所以在实际使用中应尽量使用layer(x)而不是使用layer.forward(x)。
Module中的可学习参数可以通过named_parameters()或者parameters()返回迭代器，前者会给每个parameter都附上名字，使其更具有辨识度
"""

# Method 1 -----------------------------------------
print("------构造模型的方法： --------------")
# Method 1 -----------------------------------------
class Net1(torch.nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 3, 1, 1)
        self.dense1 = torch.nn.Linear(32 * 3 * 3, 128)
        self.dense2 = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv(x)), 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.dense1(x))
        x = self.dense2()
        return x
		
# Method 2 ------------------------------------------
class Net2(torch.nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2))
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(32 * 3 * 3, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 10)
        )

    def forward(self, x):
        conv_out = self.conv1(x)
        res = conv_out.view(conv_out.size(0), -1)
        out = self.dense(res)
        return out

# Method 3 -------------------------------
class Net3(torch.nn.Module):
    def __init__(self):
        super(Net3, self).__init__()
        self.conv=torch.nn.Sequential()
        self.conv.add_module("conv1",torch.nn.Conv2d(3, 32, 3, 1, 1))
        self.conv.add_module("relu1",torch.nn.ReLU())
        self.conv.add_module("pool1",torch.nn.MaxPool2d(2))
        self.dense = torch.nn.Sequential()
        self.dense.add_module("dense1",torch.nn.Linear(32 * 3 * 3, 128))
        self.dense.add_module("relu2",torch.nn.ReLU())
        self.dense.add_module("dense2",torch.nn.Linear(128, 10))

    def forward(self, x):
        conv_out = self.conv1(x)
        res = conv_out.view(conv_out.size(0), -1)
        out = self.dense(res)
        return out

# Method 4 ------------------------------------------
class Net4(torch.nn.Module):
    def __init__(self):
        super(Net4, self).__init__()
        self.conv = torch.nn.Sequential(
            OrderedDict(
                [
                    ("conv1", torch.nn.Conv2d(3, 32, 3, 1, 1)),
                    ("relu1", torch.nn.ReLU()),
                    ("pool", torch.nn.MaxPool2d(2))
                ]
            ))

        self.dense = torch.nn.Sequential(
            OrderedDict([
                ("dense1", torch.nn.Linear(32 * 3 * 3, 128)),
                ("relu2", torch.nn.ReLU()),
                ("dense2", torch.nn.Linear(128, 10))
            ])
        )

    def forward(self, x):
        conv_out = self.conv1(x)
        res = conv_out.view(conv_out.size(0), -1)
        out = self.dense(res)
        return out
		
# 关于nn.Module模块和nn.functional的区别：
#用nn.Module实现的layer是特殊的一个类，会自动提取可学习的参数；nn.functional的函数时一个纯函数，如果模型有参数，需要自定义
#用法：如果模型有可学习的参数，最好用前者，否则两者都可以用，另外dropout没有参数也要使用前者，因为其在训练中和测试中还是有区别，二nn.Module可以使用model.eval加以区分；




		

print("------torch维度变换的相关： --------------")		
# 新增或者压缩维度 unsqueeze(dim = 0) squeeze(dim = 0)
# 修改尺寸 view()--共享内存  
		#  resize() 与view不同，她可以修改尺寸，如果新尺寸超过了原尺寸会自动分配新的空间，如果小于，则之前的数据依旧会保存
#通道掉换 t() 转置  torch.permute()
x = torch.randn(2,3,4)
x.permute(2,0,l)  # x.size()  4*2*3

print("------torch广播法则： --------------")
#expand() expand_as() 实现的功能是重复数组，与repeat类似功能，但是repeat会复制数据多分，会额外占用空间，expand不会额外占用空间，只会在需要时才扩充
>>> x = torch.tensor([[1], [2], [3]])
>>> x.size()
torch.Size([3, 1])
>>> x.expand(3, 4)
tensor([[ 1,  1,  1,  1],
        [ 2,  2,  2,  2],
        [ 3,  3,  3,  3]])
>>> x.expand(-1, 4)   # -1 means not changing the size of that dimension
tensor([[ 1,  1,  1,  1],
        [ 2,  2,  2,  2],
        [ 3,  3,  3,  3]])
		
print("------torch常用的选择函数： --------------")
#index_select(input, dim, index) 在指定的dim维度上选取
>>> x = torch.randn(3, 4)
>>> x
tensor([[ 0.1427,  0.0231, -0.5414, -1.0009],
        [-0.4664,  0.2647, -0.1228, -1.1068],
        [-1.1734, -0.6571,  0.7230, -0.6004]])
>>> indices = torch.tensor([0, 2])
>>> torch.index_select(x, 0, indices)
tensor([[ 0.1427,  0.0231, -0.5414, -1.0009],
        [-1.1734, -0.6571,  0.7230, -0.6004]])
>>> torch.index_select(x, 1, indices)
tensor([[ 0.1427, -0.5414],
        [-0.4664, -0.1228],
        [-1.1734,  0.7230]])
		
#masked_select(input, mask) 相当于a[a>0], 使用ByteTensor进行选取
masked = torch.ByteTensot([[1,0,1],[0,0,1]])
input = torch.LongTensor([[23,34,5],[23,45,56]])
torch.masked_select(input, masked, out=out)# 23,5,56  3*1

#non_zero(input) 非0的元素

#gather(input, dim, index)
b = torch.Tensor([[1,2,3],[4,5,6]])
index_1 = torch.LongTensor([[0,1],[2,0]])
index_2 = torch.LongTensor([[0,1,1],[0,0,0]])
print torch.gather(b, dim=1, index=index_1)  # dim=1,列索引有012  解释：取某行中的第几列
print torch.gather(b, dim=0, index=index_2)  # dim=0, 行索引有01
1 2 3
4 5 6
res:
1  2
 6  4
[torch.FloatTensor of size 2x2] #输出的size与index一样
1  5  6
 1  2  3
[torch.FloatTensor of size 2x3] #输出的size与index一样

#torch.unfold()

dim (int) – dimension in which unfolding happens  维度
size (int) – the size of each slice that is unfolded 每次获取的连续步长
step (int) – the step between each slice  间隔
用法类似于arange(1, 4, 2)

>>> x = torch.arange(1., 8)
>>> x
tensor([ 1.,  2.,  3.,  4.,  5.,  6.,  7.])
>>> x.unfold(0, 2, 1)  0维度 连续取两个，取完间隔为1（步长为1 ）
tensor([[ 1.,  2.],
        [ 2.,  3.],
        [ 3.,  4.],
        [ 4.,  5.],
        [ 5.,  6.],
        [ 6.,  7.]])
>>> x.unfold(0, 2, 2)
tensor([[ 1.,  2.],
        [ 3.,  4.],
        [ 5.,  6.]])
		
print("------torch常用的逐元素操作：加减乘除指数平方求余激活函数等 --------------")
#clamp(input, min, max) #超过范围的截断
#torch.mul()点乘 和 torch.mm()矩阵相乘的区别
#torch.mul(input, value, out=None) 就是*， value可以是一个value，或者是一个相同维度的tensor
>>> a = torch.randn(3)
>>> a
tensor([ 0.2015, -0.4255,  2.6087])
>>> torch.mul(a, 100)
tensor([  20.1494,  -42.5491,  260.8663])

data = [[1,2], [3,4], [5, 6]]
tensor = torch.FloatTensor(data)
tensor.mul(tensor)
out:
tensor([[  1.,   4.],
        [  9.,  16.],
        [ 25.,  36.]])
		

#torch.mm(mat1, mat2, out=None) 矩阵相乘，x.mm(y) ， 矩阵大小需满足： (i, n)x(n, j), 类似torch.matmul()
>>> mat1 = torch.randn(2, 3)
>>> mat2 = torch.randn(3, 3)
>>> torch.mm(mat1, mat2)
tensor([[ 0.4851,  0.5037, -0.3633],
        [-0.0760, -3.6705,  2.4784]])
		
		
print("------torch常用的归并操作：均值、和等 --------------")
#mean/sum/median/mode
#norm/dist 范数/距离
#std/var 标准差、方差
#cumsum/cumprod 累加、累乘

#torch.sum(input, dim, out=None) → Tensor
>>> a = torch.randn(4, 4)
>>> a

-0.4640  0.0609  0.1122  0.4784
-1.3063  1.6443  0.4714 -0.7396
-1.3561 -0.1959  1.0609 -1.9855
 2.6833  0.5746 -0.5709 -0.4430
[torch.FloatTensor of size 4x4]

>>> torch.sum(a, 1)

 0.1874
 0.0698
-2.4767
 2.2440
[torch.FloatTensor of size 4x1]

#torch.max(input, dim, max=None, max_indices=None) -> (Tensor, LongTensor) 返回的是元组，第一维是值，第二位是索引
>> a = torch.randn(4, 4)
>> a

0.0692  0.3142  1.2513 -0.5428
0.9288  0.8552 -0.2073  0.6409
1.0695 -0.0101 -2.4507 -1.2230
0.7426 -0.7666  0.4862 -0.6628
torch.FloatTensor of size 4x4]

>>> torch.max(a, 1)
(
 1.2513
 0.9288
 1.0695
 0.7426
[torch.FloatTensor of size 4x1]
,
 2
 0
 0
 0
[torch.LongTensor of size 4x1]
)

#对于最后的输出形状：假设输入形状为（m, n, k）
# 指定dim = 0 输出为(1, n, k) 或者(n, k)
# 指定dim = 1 输出为(m, 1, k) 或者(m, k)
# 指定dim = 2 输出为(m, n, 1) 或者(m, n)


print("------torch.sort --------------")
#torch.sort(input, dim=None, descending=False, out=None) -> (Tensor, LongTensor) #返回值和索引
#对输入张量input沿着指定维按升序排序。如果不给定dim，则默认为输入的最后一维。如果指定参数descending为True，则按降序排序
>>> x = torch.randn(3, 4)
>>> sorted, indices = torch.sort(x)
>>> sorted, indices = torch.sort(x)
>>> sorted

-1.6747  0.0610  0.1190  1.4137
-1.4782  0.7159  1.0341  1.3678
-0.3324 -0.0782  0.3518  0.4763
[torch.FloatTensor of size 3x4]

>>> indices

 0  1  3  2
 2  1  0  3
 3  1  0  2
[torch.LongTensor of size 3x4]

>>> sorted, indices = torch.sort(x, 0)
>>> sorted

-1.6747 -0.0782 -1.4782 -0.3324
 0.3518  0.0610  0.4763  0.1190
 1.0341  0.7159  1.4137  1.3678
[torch.FloatTensor of size 3x4]

>>> indices

 0  2  1  2
 2  0  2  0
 1  1  0  1
[torch.LongTensor of size 3x4]

#torch.topk(input, k, dim=None, largest=True, sorted=True, out=None) -> (Tensor, LongTensor)
#沿给定dim维度返回输入张量input中 k 个最大值。 如果不指定dim，则默认为input的最后一维。 如果为largest为 False ，则返回最小的 k 个值。
>>> x = torch.arange(1, 6)
>>> torch.topk(x, 3)
(5,4,3
[torch.FloatTensor of size 3]
,
 4,3,2
[torch.LongTensor of size 3]
)
>>> torch.topk(x, 3, 0, largest=False)
(1,2,3
[torch.FloatTensor of size 3]
,
 0,1,2
[torch.LongTensor of size 3]
)

print("------pytorch 0.4 变化 --------------")
#1、Tensor现在默认requires_grad=False的Variable了. torch.Tensor和torch.autograd.Variable现在其实是同一个类! 没有本质的区别! 
#所以也就是说, 现在已经没有纯粹的Tensor了, 是个Tensor, 它就支持自动求导! 你现在要不要给Tensor包一下Variable, 都没有任何意义了
#2、使用.isinstance()或是x.type(), 用type()不能看tensor的具体类型.
print(x.type())  # OK: 'torch.DoubleTensor'
print(type(x))   # "<class 'torch.Tensor'>"
#3、scalar的支持，scalar是0-维度的Tensor
#取得一个tensor的值(返回number), 用.item()；创建scalar的话,需要用torch.tensor(number)；torch.tensor(list)也可以进行创建tensor
# 4、不要显示的指定是gpu, cpu之类的. 利用.to()来执行
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
input = data.to(device)
model = MyModule(...).to(device)
#5、torch.tensor来创建Tensor 之前是大写的
#6、orch.*like以及torch.new_*
torch.zeros_like(x) #第一个是可以创建, shape相同, 数据类型相同
x.new_ones(2) # 属性一致， 得到属性与前者相同的Tensor, 但是shape不想要一致:
x.new_ones(4, dtype=torch.int) # 也可以自定义


print("------numpy.concatenate()和torch.view() --------------")
#多维
a = np.arange(6).reshape(3,2,1)
b = np.arange(4).reshape(2,2,1)
c = np.concatenate((a,b), axis=0)  # => 5*2*1
c = np.concatenate((a,b), axis=1)  # err
c = np.concatenate((a,b), axis=2)  # err
#总结：假设从低维到高维分别是3,2,1，则两个arrary合并是时候高维部分必须相同，否则报错，而且对于多维的合并只能合并其中一维；
# 低维
a = a.reshape(6,1)
b = b.reshae(4,1)
c = np.concatenate((a,b), axis=0)  # => 10*1
c = np.concatenate((a,b), axis=1)  # err

b = b.reshae(4,1)
d = arange.reshae(4,1)
c = np.concatenate((b,d), axis=0)  # => 8*1
c = np.concatenate((b,d), axis=1)  # => 4*2
#总结：二维情况下，两个arrary合并是时候高维部分必须相同，若维度相同可合并两次，那个维度合并则那个维度相加

#对于torch.view()用法一样
a = torch.from_numpy(a) # shape 6*1
b = torch.from_numpy(b) # shape 4*1
c = torch.cat([a,b], dim=0) # => 10*1
c = torch.cat([a,b], dim=1) # err

print("------np.hstack()和np.vstack() --------------")
a=[1,2,3]
b=[4,5,6]
print(np.hstack((a,b))) # => [1 2 3 4 5 6 ] shape 1* 6
print(np.vstack((a,b))) # => shape 2*3
# 总结： np.vstack() 可理解为 np.concatenate((a,b), axis=0)， np.hstack() 可理解为 np.concatenate((a,b), axis=1)

print("------matplotlib的使用 --------------")
#显示一幅图
img = Image.open(os.path.join('images', '2007_000648' + '.jpg'))
plt.figure("Image") # 图像窗口名称
plt.imshow(img)
plt.axis('on') # 关掉坐标轴为 off
plt.title('image') # 图像题目
plt.show()

#显示多幅图
plt.figure(figsize=(10,5)) #设置窗口大小
plt.suptitle('Multi_Image') # 图片名称
plt.subplot(2,3,1), plt.title('image')
plt.imshow(img), plt.axis('off')
plt.subplot(2,3,2), plt.title('gray')
plt.imshow(gray,cmap='gray'), plt.axis('off') #这里显示灰度图要加cmap
plt.subplot(2,3,3), plt.title('img_merged')
plt.imshow(img_merged), plt.axis('off')
plt.subplot(2,3,4), plt.title('r')
plt.imshow(r,cmap='gray'), plt.axis('off')
plt.subplot(2,3,5), plt.title('g')
plt.imshow(g,cmap='gray'), plt.axis('off')
plt.subplot(2,3,6), plt.title('b')
plt.imshow(b,cmap='gray'), plt.axis('off')

plt.show()

# collection apple
import numpy as np
def collectMaxAppele(m, n):
	a = np.array([[1,3,5],[4,6,7],[2,4,9]])
	sum = np.zeros([3,3])
	
	for i in range(a.shape[0]):
		for j in range(a.shape[1]):
			if (i==0 and j==0):
				sum[0,0] = a[0,0]
				continue
			up_sum = sum[i-1, j] if (i>0) else 0
			left_sum = sum[i, j-1] if (j>0) else 0
			sum[i,j] = max([up_sum, left_sum]) + a[i, j]
	print (sum[m-1][n-1])
	print (sum)
	
# sum[N] = min(sum[N-3],sum[N-1],sum[N-5]) + 1
def maxCount(m):
	sum = [0,1,2,1,2,1]
	if m < len(sum):
		return sum[m]
	for i in range(5, m+1):
		count = min([sum[i-3],sum[i-1],sum[i-5]]) + 1
		sum.append(count)
	return sum[m]

def maxCountOrder(m):
	sum = [0,1,2,1,2,1]
	if m < len(sum):
		return sum[m]
	
	return sum[m]

print("------ contiguous()--------------")
#如果在view之前用了transpose, permute等，需要用contiguous()来返回一个contiguous copy。 
#因为view需要tensor的内存是连续的 所以x.contiguous().view()   contiguous()返回一个连续的内存拷贝
#在pytorch的最新版本0.4版本中，增加了torch.reshape(), 这与 numpy.reshape 的功能类似。它大致相当于 tensor.contiguous().view()

print("------ --------------")
   


		
构造函数保护或私有；
声明一个静态对象指针；
声明一个公共返回实例接口；
class Single
{
	protected:
		Single() {
			pthread_mutex_init(&mutex);
		}
	private:
		static pthread_mutex_t mutex;
		static Single* p;
	public:
		static Single* getInstance();
}
Single* Single::mutex;
Single* Single::p = NULL;
Single* Single::getInstance()
{
	if (p == NULL)
	{
		pthread_mutex_lock(&mutex);
		if (p==NULL)
			p = new Single();
		pthread_mutex_unlock(&mutex);
	}
	return p;
}

class Single
{
	protected:
		Single() {
			pthread_mutex_init(&mutex);
		}
	private:
		static pthread_mutex_t mutex;
	public:
		static Single* getInstance();
}
Single* Single::mutex;
Single* Single::getInstance()
{
	pthread_mutex_lock(&mutex);
	static Single obj;
	pthread_mutex_unlock(&mutex);
	return &obj;
}



