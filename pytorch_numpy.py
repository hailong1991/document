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

3、python的一些技巧

print("------列表翻倍 --------------")
a = [1, 2, 3, 4]
a = [2*i for i in a]

print("------初始化列表 --------------")
a = [0] * 10

#注意：如果你列表包含了列表，这样做会产生浅拷贝。
bag_of_bags = [[0]] * 5 # [[0], [0], [0], [0], [0]]  
bag_of_bags[0][0] = 1 # [[1], [1], [1], [1], [1]]
修改：
bag_of_bags = [[0] for _ in range(5)]  
# [[0], [0], [0], [0], [0]]
bag_of_bags[0][0] = 1  
# [[1], [0], [0], [0], [0]]

print("------构造字符串 --------------")
name = "hell"
age = 23
string = "he is" + name + "and old is " + age
修改：
string = "he is {0} and old is {1}".format(name, age)

print("------访问字典 --------------")
#如果你试图访问一个不存在的于dict的key，可能会为了避免KeyError错误，你会倾向于这样做：
#统计数字出现的次数
countr = {}  
bag = [2, 3, 1, 2, 5, 6, 7, 9, 2, 7]  
for i in bag:  
    if i in countr:
        countr[i] += 1
    else:
        countr[i] = 1

for i in range(10):  
    if i in countr:
        print("Count of {}: {}".format(i, countr[i]))
    else:
        print("Count of {}: {}".format(i, 0))

#修改：dict.get(key, default=None) key -- 字典中要查找的键, default -- 如果指定键的值不存在时，返回该默认值值。
countr = {}  
bag = [2, 3, 1, 2, 5, 6, 7, 9, 2, 7]  
for i in bag:  
    countr[i] = bag.get(i, 0) + 1 
for i in range(10):  
    print("Count of {}: {}".format(i, countr.get(i, 0)))
	
#其他方法， 更费开销：
bag = [2, 3, 1, 2, 5, 6, 7, 9, 2, 7]  
countr = dict([(num, bag.count(num)) for num in bag])
for i in range(10):  
    print("Count of {}: {}".format(i, countr.get(i, 0))

或者 用dict推导式
countr = {num: bag.count(num) for num in bag}

# 推导式总结：
列表推导式：li=[i*2 for i in range(10) if i % 2 == 0]
mca={"a":1, "b":2, "c":3, "d":4}#快速兑换字典键—值
字典推导式：dict = {v:k  for k,v in mca.items()}
#集合和列表的区别：集合是一种无重复无序的序列，1.不使用中括号，使用大括号；2.结果中无重复；3.结果是一个set()集合，集合里面是一个序列
集合推导式：squared = {i**2 for i in [1,2,3,4,5]}

print("------使用库 --------------")
from collections import Counter  
bag = [2, 3, 1, 2, 5, 6, 7, 9, 2, 7]  
countr = Counter(bag)
for i in range(10):  
    print("Count of {}: {}".format(i, countr[i]))

print("------在列表中切片/步进 --------------")
a = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
#获取倒数五个 a[-5:]
#间隔两步获取 b = a[::2]
bag = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  
for elem in bag[::2]:   #list[::2]意思是遍历列表同时两步取出一个元素
    print(elem)
# 或者用 ranges
bag = list(range(0,10,2))  
print(bag)

print("------range和arange --------------")
for i in arange(0,10,2):  # err 
	pass
#arange是numpy的接口
range(1,10,2) # [1, 3, 5, 7, 9]
np.arange(1,10,2)  # array([1, 3, 5, 7, 9])
range(1,5,0.5) # err
np.arange(1,5,0.5) # array([ 1. ,  1.5,  2. ,  2.5,  3. ,  3.5,  4. ,  4.5])





