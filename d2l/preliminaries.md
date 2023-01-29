# 预备知识

## 1. 数据操作

### 1.1 入门

* 使用 **arange** 创建一个行向量 x。

  ```python
  x = x.arange(12)
  ```

  ```
  tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
  ```

* 通过 **shape** 属性访问张量的形状。

  ```python
  x.shape
  ```

  ```
  torch.Size([12])
  ```

* 调用 **numel()** 获得张量中元素的总数，即形状的所有元素乘积。

  ```python
  x.numel()
  ```

  ```
  12
  ```

* 调用 **reshape** 函数，改变一个张量的形状而不改变元素数量和元素值。

  ```python
  X = x.reshape(3, 4)
  ```

  ```
  tensor([[0, 1, 2, 3],
  		[4, 5, 6, 7],
  		[8, 9, 10, 11]])
  ```

  可以通过参数 -1 来自动计算维度。例如：可以用 `x.reshape(-1, 4)`或 `x.reshape(3, -1)`来取代`x.reshape(3, 4)`。

* 调用 **torch.zeros()** 创建全0张量。

  ```python
  torch.zeros((2, 3, 4))
  ```

  ```
  tensor([[[0., 0., 0., 0.],
  		 [0., 0., 0., 0.],
  		 [0., 0., 0., 0.]],
  		 
  		[[0., 0., 0., 0.],
  		 [0., 0., 0., 0.],
  		 [0., 0., 0., 0.]]])
  ```

* 调用 **torch.ones()** 创建全1张量。

  ```python
  torch.ones((2, 3, 4))
  ```

   ```
   tensor([[[1., 1., 1., 1.],
   		 [1., 1., 1., 1.],
   		 [1., 1., 1., 1.]],
   		 
   		[[1., 1., 1., 1.],
   		 [1., 1., 1., 1.],
   		 [1., 1., 1., 1.]]])
   ```

* 调用 **torch.randn()** 创建一个张量，其中的每个元素都从均值为0、标准差为1的标准高斯分布（正态分布）中随机采样。

  ```python
  torch.randn(3,4)
  ```

  ```
  tensor([[ 0.7277, -1.3848, -0.2607,  0.9701],
          [-2.3290, -0.3754,  0.2457,  0.0760],
          [-1.2832, -0.3600, -0.3321,  0.8184]])
  ```

* 通过提供包含数值的Python列表来为所需张量中的每个元素赋予确定值。

  ```python
  torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
  ```

  ```
  tensor([[2, 1, 4, 3],
          [1, 2, 3, 4],
          [4, 3, 2, 1]])
  ```

### 1.2 运算符

* 按元素运算：

  ```python
  x + y, x - y, x * y, x / y, x ** y  # **运算符是求幂运算
  ```

  ```python
  torch.exp(x)
  ```

* 调用 **torch.cat()** 连结多个张量，dim 参数决定沿哪个轴连结：

  ```python
  X = torch.arange(12, dtype=torch.float32).reshape((3,4))
  Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
  torch.cat((X, Y), dim=0), torch.cat((X, Y), dim=1)
  ```

  ```
  (tensor([[ 0.,  1.,  2.,  3.],
           [ 4.,  5.,  6.,  7.],
           [ 8.,  9., 10., 11.],
           [ 2.,  1.,  4.,  3.],
           [ 1.,  2.,  3.,  4.],
           [ 4.,  3.,  2.,  1.]]),
   tensor([[ 0.,  1.,  2.,  3.,  2.,  1.,  4.,  3.],
           [ 4.,  5.,  6.,  7.,  1.,  2.,  3.,  4.],
           [ 8.,  9., 10., 11.,  4.,  3.,  2.,  1.]]))
  ```

* 通过逻辑运算符构建二元张量。以 `X==Y` 为例：

  ```python
  X == Y
  ```

  ```
  tensor([[False,  True, False,  True],
          [False, False, False, False],
          [False, False, False, False]])
  ```

  **X<Y和X>Y** 同理。

* 调用 **sum()** 对张量中的所有元素进行求和，会产生一个单元素张量：

  ```python
  X.sum()
  ```

  ```
  tensor(66.)
  ```

### 1.3 广播机制

在某些情况下，即使形状不同，我们仍然可以通过调用 **广播机制** 来执行按元素操作。

```python
a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
```

```
(tensor([[0],
		[1],
		[2]]),
 tensor([[0, 1]]))
```

```python
a + b
```

```
tensor([[0, 1],
	    [1, 2],
	    [2, 3]])
```

### 1.4 索引和切片

张量与任何Python数组一样：第一个元素的索引是0，最后一个元素索引是-1。

除读取外，还可以通过指定索引将元素写入矩阵：

```python
X[1, 2] = 9
```

```
tensor([[ 0.,  1.,  2.,  3.],
        [ 4.,  5.,  9.,  7.],
        [ 8.,  9., 10., 11.]])
```

还可以为多个元素赋值相同的值：

```python
X[0:2, :] = 12
```

```
tensor([[12., 12., 12., 12.],
        [12., 12., 12., 12.],
        [ 8.,  9., 10., 11.]])
```

### 1.5 节省内存

运行一些操作可能会导致为新结果分配内存。例如：如果我们用 **Y = X + Y** ，我们将取消引用Y指向的张量，而是指向新分配的内存处的张量。

Python 的 **id()** 函数给我们提供了内存中引用对象的确切地址。

```python
before = id(Y)
Y = Y + X
id(Y) == before
```

```python
False
```

正确方法：**X[ : ] = X + Y 或 X += Y**

```python
before = id(X)
X += Y
id(X) == before
```

```python
True
```

### 1.6 转换为其他Python对象

* torch张量与NumPy张量互相转换：

  ```python
  A = X.numpy()
  B = torch.tensor(A)
  type(A), type(B)
  ```

  ```
  (numpy.ndarray, torch.Tensor)
  ```

* 调用**item函数或Python内置函数**，将大小为1的张量转换为 Python 标量：

  ```python
  a = torch.tensor([3.5])
  a, a.item(), float(a), int(a)
  ```

  ```
  (tensor([3.5000]), 3.5, 3.5 ,3)
  ```



## 2. 数据预处理

### 2.1 读取数据集

* `os.path.join()`用于路径拼接。

  * 不存在以 “/" 开始的参数，则在首个参数前加上"/"；
  * 存在以 "/" 开始的参数，从最后一个以 "/" 开头的参数开始拼接，前面的参数全部丢弃；
  * 同时存在以 "./" 和 "/" 开始的参数，则以 "/" 为主；
  * 只存在以 "./" 开始的参数，则从 "./" 开头的参数的上一个参数开始拼接。

* `os.makedirs(path, mode=511, exist_ok=False)` 方法用于递归创建多层目录。

  * **path** -- 需要递归创建的目录，可以是相对或者绝对路径；
  * **mode** -- 权限模式，默认的模式为 511 (八进制)；
  * **exist_ok**：是否在目录存在时触发异常。如果 exist_ok 为 False（默认值），则在目标目录已存在的情况下触发 FileExistsError 异常；如果 exist_ok 为 True，则在目标目录已存在的情况下不会触发 FileExistsError 异常。

  ```python
  import os
  
  os.makedirs(os.path.join('..', 'data'), exist_ok=True)
  data_file = os.path.join('..', 'data', 'house_tiny.csv')
  with open(data_file, 'w') as f:
      f.write('NumRooms,Alley,Price\n')  # 列名
      f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
      f.write('2,NA,106000\n')
      f.write('4,NA,178100\n')
      f.write('NA,NA,140000\n')
  ```

* 导入 **pandas** 包并调用 **read_csv** 函数，从CSV文件中加载原始数据集：

  ```python
  # 如果没有安装pandas，只需取消对以下行的注释来安装pandas
  # !pip install pandas
  import pandas as pd
  
  data = pd.read_csv(data_file)
  print(data)
  ```

  ```
     NumRooms Alley   Price
  0       NaN  Pave  127500
  1       2.0   NaN  106000
  2       4.0   NaN  178100
  3       NaN   NaN  140000
  ```

### 2.2 处理缺失值

“NaN”项代表缺失值。处理缺失值典型的方法包括 **插值法** 和 **删除法**。

* 插值法：通过位置索引 **iloc**，将**data**分为 **inputs** 和 **outputs**，对于 **inputs** 中缺少的数值，我们用同一列的均值替换 "NAN" 项：

  ```python
  inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
  inputs = inputs.fillna(inputs.mean())
  print(inputs)
  ```

  ```
     NumRooms Alley
  0       3.0  Pave
  1       2.0   NaN
  2       4.0   NaN
  3       3.0   NaN
  ```

* **get_dummies( data, prefix=None, prefix_sep='_', dummy_na=False, columns=None, sparse=False, drop_first=False, dtype=None)** 是 **pandas** 实现 **one hot encode** 的方式。

  ```as3
  data : array-like, Series, or DataFrame
  输入的数据
  prefix : string, list of strings, or dict of strings, default None
  get_dummies转换后，列名的前缀
  columns : list-like, default None
  指定需要实现类别转换的列名
  dummy_na : bool, default False
  增加一列表示空缺值，如果False就忽略空缺值
  drop_first : bool, default False
  获得k中的k-1个类别值，去除第一个
  ```

  ```python
  inputs = pd.get_dummies(inputs, dummy_na=True)
  print(inputs)
  ```

  ```
     NumRooms  Alley_Pave  Alley_nan
  0       3.0           1          0
  1       2.0           0          1
  2       4.0           0          1
  3       3.0           0          1
  ```

### 2.3 转换为张量格式

```python
import torch

X, y = torch.tensor(inputs.values), torch.tensor(outputs.values)
X, y
```

```
(tensor([[3., 1., 0.],
         [2., 0., 1.],
         [4., 0., 1.],
         [3., 0., 1.]], dtype=torch.float64),
 tensor([127500, 106000, 178100, 140000]))
```



## 3. 线性代数

### 3.1 标量

### 3.2 向量

#### 3.2.1 长度、维度和形状

可以通过调用 Python 的内置 **len()函数** 来访问张量的长度：

```python
x = torch.arange(4)
len(x)
```

```
4
```

明确本书中维度的定义：向量或轴的维度被用来表示向量或轴的长度，即向量或轴的元素数量。 然而，张量的维度用来表示张量具有的轴数。 在这个意义上，张量的某个轴的维数就是这个轴的长度。

### 3.3 矩阵

```python
A = torch.arange(20).reshape(5, 4)
A
```

```
tensor([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11],
        [12, 13, 14, 15],
        [16, 17, 18, 19]])
```

* 转置

  ```python
  A.T
  ```

  ```
  tensor([[ 0,  4,  8, 12, 16],
          [ 1,  5,  9, 13, 17],
          [ 2,  6, 10, 14, 18],
          [ 3,  7, 11, 15, 19]])
  ```

### 3.4 张量

张量是描述具有任意数字轴的n维数组的通用方法。

```python
X = torch.arange(24).reshape(2, 3, 4)
X
```

```
tensor([[[ 0,  1,  2,  3],
         [ 4,  5,  6,  7],
         [ 8,  9, 10, 11]],

        [[12, 13, 14, 15],
         [16, 17, 18, 19],
         [20, 21, 22, 23]]])
```

### 3.5 张量算法的基本性质

* 按元素操作：

  * 按元素加法：

    ```python
    A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    B = A.clone()  # 通过分配新内存，将A的一个副本分配给B
    A, A + B
    ```

    ```
    (tensor([[ 0.,  1.,  2.,  3.],
             [ 4.,  5.,  6.,  7.],
             [ 8.,  9., 10., 11.],
             [12., 13., 14., 15.],
             [16., 17., 18., 19.]]),
     tensor([[ 0.,  2.,  4.,  6.],
             [ 8., 10., 12., 14.],
             [16., 18., 20., 22.],
             [24., 26., 28., 30.],
             [32., 34., 36., 38.]]))
    ```

  * 按元素乘法：称为 **Hadamard积**，（数学符号⊙）：

    ```python
    A * B
    ```

    ```
    tensor([[  0.,   1.,   4.,   9.],
            [ 16.,  25.,  36.,  49.],
            [ 64.,  81., 100., 121.],
            [144., 169., 196., 225.],
            [256., 289., 324., 361.]])
    ```

  * 张量乘以或加上一个标量，其中张量的每个元素都将与标量相加或相乘：

    ```python
    a = 2
    X = torch.arange(24).reshape(2, 3, 4)
    a + X, (a * X).shape
    ```

    ```
    (tensor([[[ 2,  3,  4,  5],
              [ 6,  7,  8,  9],
              [10, 11, 12, 13]],
    
             [[14, 15, 16, 17],
              [18, 19, 20, 21],
              [22, 23, 24, 25]]]),
     torch.Size([2, 3, 4]))
    ```

### 3.6 降维

* 求和函数 **sum()**：

  ```python
  x = torch.arange(4, dtype=torch.float32)
  x, x.sum()
  ```

  ```
  (tensor([0., 1., 2., 3.]), tensor(6.))
  ```

  * 默认情况下，调用求和函数会沿所有的轴降低张量的维度，使它变成一个标量：

    ```python
    A.shape, A.sum()
    ```

    ```
    torch.Size([5, 4]), tensor(190.))
    ```

  * 还可以指定张量沿哪一个轴来通过求和降低维度：

    ```python
    A_sum_axis0 = A.sum(axis=0)
    A_sum_axis0, A_sum_axis0.shape
    ```

    ```
    (tensor([40., 45., 50., 55.]), torch.Size([4]))
    ```

    ```python
    A_sum_axis1 = A.sum(axis=1)
    A_sum_axis1, A_sum_axis1.shape
    ```

    ```
    (tensor([ 6., 22., 38., 54., 70.]), torch.Size([5]))
    ```

  * 还可以沿多个轴求和：

    ```python
    A.sum(axis=[0, 1])  # 结果和A.sum()相同
    ```

    ```
    tensor(190.)
    ```

* **mean()** 求平均值：

  ```python
  A.mean(), A.sum() / A.numel()
  ```

  ```
  (tensor(9.5000), tensor(9.5000))
  ```

  * 同样，计算平均值的函数也可以沿指定轴降低张量的维度：

    ```python
    A.mean(axis=0), A.sum(axis=0) / A.shape[0]
    ```

    ```
    (tensor([ 8.,  9., 10., 11.]), tensor([ 8.,  9., 10., 11.]))
    ```

#### 3.6.1 非降维求和

* 利用 **sum()** 和 **mean()** 的 **keepdims** 参数可以保持求和或计算均值时保持轴数不变：

  ```python
  sum_A = A.sum(axis=1, keepdims=True)
  sum_A
  ```

  ```
  tensor([[ 6.],
          [22.],
          [38.],
          [54.],
          [70.]])
  ```

* 如果我们想沿某个轴计算A元素的累计总和，可以调用 **cumsum** 函数：

  ```python
  A.cumsum(axis=0)
  ```

  ```
  tensor([[ 0.,  1.,  2.,  3.],
          [ 4.,  6.,  8., 10.],
          [12., 15., 18., 21.],
          [24., 28., 32., 36.],
          [40., 45., 50., 55.]])
  ```

### 3.7 点积

给定两个向量 $x,y\in R^d$，它们的点积 $x^Ty(或<x,y>)$ 是相同位置的按元素乘积的和：  $x^Ty=\sum_{i=1}^{d}x_iy_i$ 。**torch.dot()**：

```python
y = torch.ones(4, dtype = torch.float32)
x, y, torch.dot(x, y)
```

```
(tensor([0., 1., 2., 3.]), tensor([1., 1., 1., 1.]), tensor(6.))
```

也可以通过执行按元素乘法，然后进行求和来表示两个向量的点积：

```python
torch.sum(x * y)
```

```
tensor(6.)
```

### 3.8 矩阵-向量积

矩阵 $A\in R^{m\times n}$，向量 $x\in R^n$，矩阵-向量积 **torch.mv()** ：
$$
A=
\begin{bmatrix}
&a_1^T&\\
&a_2^T&\\
&\vdots&\\
&a_m^T&
\end{bmatrix}
$$

$$
Ax=
\begin{bmatrix}
&a_1^T&\\
&a_2^T&\\
&\vdots&\\
&a_m^T&
\end{bmatrix}
x=
\begin{bmatrix}
&a_1^Tx&\\
&a_2^Tx&\\
&\vdots&\\
&a_m^Tx&
\end{bmatrix}
$$



```python
A.shape, x.shape, torch.mv(A, x)
```

```
(torch.Size([5, 4]), torch.Size([4]), tensor([ 14.,  38.,  62.,  86., 110.]))
```

### 3.9 矩阵-矩阵乘法

矩阵-矩阵乘法 **torch.mm()** ：

```python
B = torch.ones(4, 3)
torch.mm(A, B)
```

```
tensor([[ 6.,  6.,  6.],
        [22., 22., 22.],
        [38., 38., 38.],
        [54., 54., 54.],
        [70., 70., 70.]])
```

### 3.10 范数

在线性代数中，向量范数是将向量映射到标量的函数 **f**。

向量范数的性质：

* $f(\alpha x)=|\alpha|f(x)$
* $f(x+y)\le f(x)+f(y)$
* $f(x)\ge 0$
* 范数最小为0，当且仅当向量全由0组成

欧几里得距离是一个 $L_2$ 范数： $||x||_2=||x||=\sqrt{\sum_{i=1}^{n}x_i^2}$ 。用 **torch.norm()** 计算向量的范数：

```python
u = torch.tensor([3.0, -4.0])
torch.norm(u)
```

```
tensor(5.)
```

$L_1$ 范数，表示为向量元素的绝对值之和。 $||x||_1=\sum_{i=1}^{n}|x_i|$ ：

```python
torch.abs(u).sum()
```

```
tensor(7.)
```

一般地，$L_p$ 范数： $||x||_p=(\sum_{i=1}^{n}|x_i|^p)^{1/p}$ 

类似于向量的 $L_2$ 范数，矩阵 $X\in R^{m\times n}$ 的Frobenius范数  $||X||_F=\sqrt{\sum_{i=1}^{m}\sum_{j=1}^{n}x_{ij}^2}$ ：

```python
torch.norm(torch.ones((4, 9)))
```



```
tensor(6.)
```

### 3.11 特殊矩阵

* 对称矩阵和反对称矩阵： $A_{ij}=A_{ji}\ and\ A_{ij}=-A_{ji}$ 
* 正定矩阵：设M是n阶方阵，如果对任何非零向量 $z$，都有 $z^TMz>0$ ，其中 $z^T$ 表示 $z$ 的转置，就称M为正定矩阵
* 正交矩阵：所有行都互相正交、所有行都有单位长度，可以写成 $UU^T=I$ 
* 置换矩阵：  $P\ where\ P_{ij}=1\ if\ and\ only\ if\ j=\pi(i)$ ，一定是正交矩阵

### 3.12 特征值和特征向量

* 特征向量：不被矩阵改变方向的向量
* 特征值：特征向量被矩阵改变的长度的倍数



## 4. 微积分

### 4.1 导数和微分

### 4.2 偏导数

### 4.3 梯度

设函数 $f:R^n\rightarrow R$ 的输入是一个n维向量 $x=[x_1,x_2,\cdots,x_n]^T$ ，并且输出是一个标量、函数 $f(x)$ 相对于 **x** 的梯度是一个包含n个偏导数的向量： $\nabla_xf(x)=[\frac{\partial f(x)}{\partial x_1},\frac{\partial f(x)}{\partial x_2},\cdots,\frac{\partial f(x)}{\partial x_n}]^T$ 。

$\frac{\partial y}{\partial x}$ ：

* y是标量，**x** 是向量： $\frac{\partial y}{\partial x}=[\frac{\partial y}{\partial x_1},\frac{\partial y}{\partial x_2},\cdots,\frac{\partial y}{\partial x_n}]$ 

* **y** 是向量，x 是标量： $\frac{\partial y}{\partial x}=[\frac{\partial y_1}{\partial x},\frac{\partial y_2}{\partial x},\cdots,\frac{\partial y_m}{\partial x}]^T$ 

* **y** 和 **x** 都是向量： 
  $$
  \frac{\partial y}{\partial x}=
  \begin{bmatrix}
  \frac{\partial y_1}{\partial x}\\
  \frac{\partial y_2}{\partial x}\\
  \vdots\\
  \frac{\partial y_m}{\partial x}
  \end{bmatrix}=
  \begin{bmatrix}
  &\frac{\partial y_1}{\partial x_1},&\frac{\partial y_1}{\partial x_2},&\cdots,&\frac{\partial y_1}{\partial x_n}\\
  &\frac{\partial y_2}{\partial x_1},&\frac{\partial y_2}{\partial x_2},&\cdots,&\frac{\partial y_2}{\partial x_n}\\
  &\vdots &\vdots &\ddots &\vdots\\
  &\frac{\partial y_m}{\partial x_1},&\frac{\partial y_m}{\partial x_2},&\cdots,&\frac{\partial y_m}{\partial x_n}
  \end{bmatrix}
  $$

微分多元函数常用规则：

* 对于所有 $A\in R^{m\times n}，都有 \nabla_xAx=A^T$
* 对于所有 $A\in R^{n\times m}，都有 \nabla_xx^TA=A$
* 对于所有 $A\in R^{n\times n},都有 \nabla_xx^TAx=(A+A^T)x$
* $\nabla_x||x||^2=\nabla_xx^Tx=2x$ 
* 同样，对于任何矩阵 $X$，都有 $\nabla_X||X||^2_F=2X$  

### 4.4 链式法则



## 5. 自动微分

### 5.1 一个简单的例子

```python
import torch

x = torch.arange(4.0)
x.requires_grad_(True)  # 等价于x=torch.arange(4.0,requires_grad=True)
y = 2 * torch.dot(x, x)
y.backward()
x.grad
```

```
tensor([ 0.,  4.,  8., 12.])
```

```python
x.grad == 4 * x
```

```
tensor([True, True, True, True])
```

```python
# 在默认情况下，PyTorch会累积梯度，我们需要清除之前的值
x.grad.zero_()
y = x.sum()
y.backward()
x.grad
```

```
tensor([1., 1., 1., 1.])
```

### 5.2 非标量变量的反向传播

```python
# 对非标量调用backward需要传入一个gradient参数，该参数指定微分函数关于self的梯度。
# 本例只想求偏导数的和，所以传递一个1的梯度是合适的
x.grad.zero_()
y = x * x
# 等价于y.backward(torch.ones(len(x)))
y.sum().backward()
x.grad
```

```
tensor([0., 2., 4., 6.])
```

### 5.3 分离计算

有时，我们希望将某些计算移动到记录的计算图之外。例如，假设`y`是作为`x`的函数计算的，而`z`则是作为`y`和`x`的函数计算的。 想象一下，我们想计算`z`关于`x`的梯度，但由于某种原因，希望将`y`视为一个常数， 并且只考虑到`x`在`y`被计算后发挥的作用。

```python
x.grad.zero_()
y = x * x
u = y.detach()
z = u * x

z.sum().backward()
x.grad == u
```

```
tensor([True, True, True, True])
```

由于记录了`y`的计算结果，我们可以随后在`y`上调用反向传播， 得到`y=x*x`关于的`x`的导数，即`2*x`。

```python
x.grad.zero_()
y.sum().backward()
x.grad == 2 * x
```

```
tensor([True, True, True, True])
```

### 5.4 Python 控制流的梯度计算

```python
def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c

a = torch.randn(size=(), requires_grad=True)
d = f(a)
d.backward()
a.grad == d / a
```

```
tensor(True)
```

