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
print("------构造字典 --------------")

plot_name = {k:v for k, v in zip(label_name, val_accuracy)}

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

#或者 用dict推导式
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


print("------type和isinstance --------------")  #用法：isinstance(object, classinfo)
#isinstance() 与 type() 区别：
#type() 不会认为子类是一种父类类型，不考虑继承关系。
#isinstance() 会认为子类是一种父类类型，考虑继承关系。
class A:
    pass
class B(A):
    pass
isinstance(A(), A)    # returns True
type(A()) == A        # returns True
isinstance(B(), A)    # returns True
type(B()) == A        # returns False

a = 2
isinstance (a,int) # True
isinstance (a,str) # False
OrderedDict

print("------dict和OrderedDict--------------")
#常规dict并不跟踪插入顺序，迭代处理会根据键在散列表中存储的顺序来生成值。在OrderDict中则相反，它会记住元素插入的顺序，并在创建迭代器时使用这个顺序
import collections
d = {}
d['a'] = 'A'
d['b'] = 'B'
d['c'] = 'C'
for k, v in d.items():
  print k, v
#无序
a A
c C
b B

d = collections.OrderedDict()
d['a'] = 'A'
d['b'] = 'B'
d['c'] = 'C'
for k, v in d.items():
  print k, v
#有序
a A
b B
c C

print("------获取dict的最大值--------------")
d = {1:1, 2:0, 3:2}
min(d) # out:1   默认比较键值
法一：min_key = min(d, key = d.get)  # d表示传入的是键，d.get是一个函数对象，得到的结果是value，然后比较函数的值，将最小值对应的d 返回
法二：min_key = min(d, key = lambda x: d[x]) # d表示传入的是键， lambda x: d[x]是一个函数对象，然后比较函数的值，将最小值对应的d 返回

min(d.items(), key=lambda x : x[1])# d.items()表示传入的是元组，lambda x : x[1]是一个函数对象，得到的元组的第二位即value，然后比较函数的值，最后d.items()最后将 返回
 #out (2, 0)
# 获取v\k
value_key = max(zip(d.values(), d.keys()))

print("------struct.pack--------------")
struct.pack(fmt,v1,v2,.....)  #将v1,v2等参数的值进行一层包装，包装的方法由fmt指定。被包装的参数必须严格符合fmt。最后返回一个包装后的字符串
struct.unpack(fmt,string) #名思义，解包。比如pack打包，然后就可以用unpack解包了。返回一个由解包数据(string)得到的一个元组(tuple), 
#即使仅有一个数据也会被解包成元组。其中len(string) 必须等于 calcsize(fmt)，这里面涉及到了一个calcsize函数。
struct.calcsize(fmt) #这个就是用来计算fmt格式所描述的结构的大小。
eg:
a = 20
b = 1
str = struct.pack("ii", a,b)
print("str len:", len(str))  # str 为二进制
print(type(str), str) 
https://www.cnblogs.com/xiao-apple36/p/9276777.html#_label2_1

print("------关于继承中父类调用子类成员问题 --------------")

class father():
    def __init__(self):
        self._name = "aaa"
 
    def fun1(self):
        method = eval('self.' + 'ss' + '_db')
        print(method)
 
    def fun2(self):
        tt = self.ss_db
        print(tt)
 
 
class son(father):
    def __init__(self):
        father.__init__(self)
        self.ss_db = "daas"
 
    def getname(self):
        print(self._name)
        return self._name
 
 
if __name__ == "__main__":
    son = son()
    son.fun1() # daas
    son.fun2() # daas
    son.getname()# aaa


#理解：python是一个动态语言，应该从对象的角度去看，只要运行期的对象有该方法就行，这里的self不是指father或者它的实例
#不用从"父类可以调用子类中的方法"这种角度去看,而从对象生成以后的执行调用这个角度去看就行了,
#对象调用的时候 self.get_request() 就是调用自己这个对象里的 get_request() 方法啊,那这个方法具体是哪个,是由对象是从哪个类产生决定的
print("------str.replace --------------")
str = "this is string example....wow!!! this is really string";
print str.replace("is", "was");
print str.replace("is", "was", 3);

print("------glob.glob() --------------")
#glob模块的主要方法就是glob,该方法返回所有匹配的文件路径列表（list）；该方法需要一个参数用来指定匹配的路径字符串（字符串可以为绝对路径也可以为相对路径），
#其返回的文件名只包括当前目录里的文件名，不包括子文件夹里的文件
glob.glob(’c:*.txt’) # 获得C盘下的所有txt文件
glob.glob(’E:\pic**.jpg’) # 我这里就是获得C盘下的所有txt文件

# iglob 获取一个迭代器（ iterator ）对象，使用它可以逐个获取匹配的文件路径名。
#与glob.glob()的区别是：glob.glob同时获取所有的匹配路径，而 glob.iglob一次只获取一个匹配路径。下面是一个简单的例子
f = glob.iglob(r'../*.py')
for py in f:
    print py # f是一个迭代器对象，通过遍历，可以输出所有满足条件的*.py文件
	
print("------ @property    @staticmethod --------------")
class Rectangle(object):
  def __init__(self):
    self.width =10
    self.height=20
r=Rectangle()
r.width=1.0 # 此时属性可以改变，不安全

class Rectangle(object):
  @property
  def width(self):
    #变量名不与方法名重复，改为true_width，下同
    return self.true_width
s = Rectangle()
print (s.width) # 可读变量
s.width = 1024 # err 会报错，@property 声明私有，此时不能改变，只读变量

class Rectangle(object):
  @property
  def width(self):
    # 变量名不与方法名重复，改为true_width，下同
    return self.true_width
  @width.setter
  def width(self, input_width):
    self.true_width = input_width
s = Rectangle()
# 与方法名一致
s.width = 1024 # 不会报错，@width.setter声明了可以改变比变量

# 声明以下接口，可对变量删除
@height.deleter
def height(self):
    del self.true_height
#总结@property声明这是私有变量，只可以读，要写或者删除需要重写@width.setter和@width.deleter

# @staticmethod 声明是静态变量或者静态函数，独立于实例，可以理解为私有函数的一个公共接口，服务于私有函数
print("------ list append()和extend()--------------")
#list.append(object) 向列表中添加一个对象object
#list.extend(sequence) 把一个序列seq的内容添加到列表中
music_media = ['compact disc', '8-track tape', 'long playing record']
new_media = ['DVD Audio disc', 'Super Audio CD']
music_media.append(new_media) # ['compact disc', '8-track tape', 'long playing record',['DVD Audio disc', 'Super Audio CD']]
music_media.extend(new_media) # ['compact disc', '8-track tape', 'long playing record','DVD Audio disc', 'Super Audio CD']


print("------ list删除元素--------------")
a = [1,2,2,5,4,2]
#remove: 删除单个元素，删除首个符合条件的元素，按值删除
a.remove(2)
#pop:  删除单个或多个元素，按位删除(根据索引删除)
a.pop(3)
#del：它是根据索引(元素所在位置)来删除
del str[1]

#举例：删除a中所有的2
#法一：b = [i for i in a if i !=2]
#法二： #err
for i in a:
	if i == 2:
		a.remove(i)
#当列表每次删除的时候，后面的索引会向前，导致有些数据会跳过。
#在上述for循环中，假设我们删除了index=2的值，原本index=3及之后的值会向前补位，所以在循环中就跳过了原index=3的变量 
#同理，使用list.pop()函数删除指定元素的时候，也会出现上述情况

#举例二：a = [1,2,0,3,0,4],将所有0移到后面，且保持其他顺序不变
def moveZeroes(self, nums):
        j=0
        for i in xrange(len(nums)):
            if nums[j] == 0:
                nums.append(nums.pop(j)) # 这里利用了nums.pop(j)删除后返回的值为删除元素的值

            else:
                j+=1

public int calMaxSumOfArray(int[] a) {
    if (null == a) {
        return 0;
    }
    if (a.length == 1) {
        return a[0];
    }
    int sum = a[0];
    int temp = a[0];
    for (int i = 1; i < a.length; i++) {
        if (temp < 0) {
            temp = 0;
        }
        temp = temp + a[i];
        if (sum < temp) {
            sum = temp;
        }
    }
    return sum;
}


