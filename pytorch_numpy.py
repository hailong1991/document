1、numpy where的用法：
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
np.where(a > 6) ==== np.where([False, False, False,True,  True]) #等价

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