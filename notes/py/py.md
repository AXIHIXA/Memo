# `Python` Notes

- [`Python 3.7.8文档`](https://docs.python.org/zh-cn/3.7/index.html)阅读笔记
- 当然只记不完全会的，所以缺东西那是正常的

### 🌱 `Python`的非正式介绍

- 数字
    - 运算符：`+`，`-`，`*`，`**`，`/`，`//`，`%`，`()`
    - 整数类型为`int`，有小数部分的数类型为`float`
    - 除法运算`/`永远返回`float`；如果需要 *地板除* （floor division）得到整数结果，需使用`//`
    - 如果一个变量 *未定义* （未赋值），使用时会报`NameError`
    - *交互模式* 下，上一次打印出来的表达式被赋值给变量`_`
        - 使用示例：继续计算
        ```
        >>> tax = 12.5 / 100
        >>> price = 100.50
        >>> price * tax
        12.5625
        >>> price + _
        113.0625
        >>> round(_, 2)
        113.06
        ```
        - 这个变量应该被使用者当作是 *只读* 类型。**不要**向它显式地赋值，否则将显示创建一个同名变量，并屏蔽掉内建的`_`
    - 除了`int`和`float`，`python`还支持其他类型的数字，例如`Decimal`，`Fraction`以及 *复数* （使用`j`或`J`表示虚数部分，例如`3 + 5j`）
- 字符串
    - 字符串使用 *单引号* `'...'`以及 *双引号* `"..."`都可以获得同样的结果
    - *反斜杠* `\` 用于 *转义* 
    - *交互模式* 中，输出的字符串外面会加上引号，特殊字符会使用反斜杠来转义 
    - 如果字符串中有单引号而没有双引号，该字符串外将加双引号来表示，否则就加单引号
    - [`print`](https://docs.python.org/zh-cn/3.8/library/functions.html#print)函数会生成可读性更强的输出，即略去两边的引号，并且打印出经过转义的特殊字符
    - *原始字符串* ：在引号前添加`r`即可忽略转义
        - 例如
        ```
        >>> print('C:\some\name')  # here \n means newline!
        C:\some
        ame
        >>> print(r'C:\some\name')  # note the r before the quote
        C:\some\name
        ```
    - 字符串字面值可以跨行连续输入
        - 一种方式是用 *三重引号* `"""..."""`或`'''...'''`
        - 字符串中的回车换行会自动包含到字符串中
        - 如果不想包含，在行尾添加一个`\`即可。如下例
        ```
        print("""\
        Usage: thingy [OPTIONS]
             -h                        Display this usage message
             -H hostname               Hostname to connect to
        """)
        ```
        - 将产生如下输出（注意最开始的换行**没有**包括进来）
        ```
        Usage: thingy [OPTIONS]
             -h                        Display this usage message
             -H hostname               Hostname to connect to
        ```
    - 字符串可以用`+`进行 *连接* （粘到一起），也可以用`*`进行 *重复* 
    ```
    >>> # 3 times 'un', followed by 'ium'
    >>> 3 * 'un' + 'ium'
    'unununium'
    ```
    - 相邻的两个或多个字符串 *字面值*  （引号引起来的字符）将会 *自动连接* 到一起
        - 例如
        ```
        >>> 'Py' 'thon'
        'Python'

        >>> text = ('Put several strings within parentheses '
        ...         'to have them joined together.')
        >>> text
        'Put several strings within parentheses to have them joined together.'
        ```
        - 只能对两个 *字面值* 这样操作， *变量* 或 *表达式* **不行**
            - 连接变量或表达式可以使用`+`
    - 字符串 *索引* （下标访问）
        - 第一个字符索引是`0`
        - *单个字符* 并**没有**特殊的类型，只是一个长度为一的字符串
        - *负数索引值* ：从`-1`开始（因为`-0`和`0`是一样的），含义为从右边开始计数
        ```
        >>> word = 'Python'
        >>> word[0]   # character in position 0
        'P'
        >>> word[5]   # character in position 5
        'n'
        >>> word[-1]  # last character
        'n'
        >>> word[-2]  # second-last character
        'o'
        >>> word[-6]
        'P'
        ```
        - 使用越界索引会产生`IndexError`
    - 字符串 *切片* 
        - 索引可以得到单个字符，而切片可以获取 *子串* 
        - 切片的索引值为左闭右开区间
        ```
        >>> word[0:2]  # characters from position 0 (included) to 2 (excluded)
        'Py'
        >>> word[2:5]  # characters from position 2 (included) to 5 (excluded)
        'tho'
        ```
        - 这使得`s[:i] + s[i:]`总是等于`s`
        ```
        >>> word[:2] + word[2:]
        'Python'
        >>> word[:4] + word[4:]
        'Python'
        ```
        - 切片索引默认值：省略开始索引时默认为0，省略结束索引时默认为到字符串的结束
        ```
        >>> word[:2]   # character from the beginning to position 2 (excluded)
        'Py'
        >>> word[4:]   # characters from position 4 (included) to the end
        'on'
        >>> word[-2:]  # characters from the second-last (included) to the end
        'on'
        ```
        - 切片中的 *越界索引* 会被自动处理，**不会**报错
        ```
        >>> word[4:42]
        'on'
        >>> word[42:]
        ''
        ```
    - `python`中的字符串**不能**被修改。向字符串的某个索引位置赋值会产生`TypeError`
        - 如果需要一个不同的字符串，应当新建一个
    ```
    >>> word[0] = 'J'
    Traceback (most recent call last):
    File "<stdin>", line 1, in <module>
    TypeError: 'str' object does not support item assignment
    >>> word[2:] = 'py'
    Traceback (most recent call last):
    File "<stdin>", line 1, in <module>
    TypeError: 'str' object does not support item assignment
    
    >>> 'J' + word[1:]
    'Jython'
    >>> word[:2] + 'py'
    'Pypy'
    ```
    - 内建函数[`len`](https://docs.python.org/zh-cn/3.8/library/functions.html#len)返回一个字符串的长度
- 列表
    - 列表可以通过方括号括起、逗号分隔的一组值（元素）得到
    - 一个列表可以包含 *不同类型* 的元素，但通常使用时各个元素类型相同
    - 和字符串（以及各种内置的`sequence`类型）一样，列表也支持 *索引* 和 *切片* 
    - 所有的切片操作都返回一个包含所请求元素的新列表
        - 这意味着以下切片操作会返回列表的一个 *浅拷贝* 
        ```
        >>> squares[:]
        [1, 4, 9, 16, 25]
        ```
        - 给切片赋值也是可以的，这样甚至可以改变列表大小，或者把列表整个清空
        ```
        >>> letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
        >>> letters
        ['a', 'b', 'c', 'd', 'e', 'f', 'g']
        >>> # replace some values
        >>> letters[2:5] = ['C', 'D', 'E']
        >>> letters
        ['a', 'b', 'C', 'D', 'E', 'f', 'g']
        >>> # now remove them
        >>> letters[2:5] = []
        >>> letters
        ['a', 'b', 'f', 'g']
        >>> # clear the list by replacing all the elements with an empty list
        >>> letters[:] = []
        >>> letters
        []
        ```
    - 列表同样支持 *拼接* 操作
    - 列表是一个`mutable`类型，就是说，它自己的内容可以改变
    - 在列表末尾通过`append`方法来添加新元素
    - 内建函数[`len`]也可以作用到列表上
    - 可以嵌套列表（创建包含其他列表的列表）

### 🌱 其他流程控制工具

#### `for`语句

- `python`中的`for`语句与你在`C`中所用到的有所不同
    - 语法
        - 其中`else: suite`部分是 *可选* 的
    ```
    for <target_list> in <expression_list>: 
        <suite> 
    else: 
        <suite> 
    ```   
    - `expression_list`会被求值一次，它应该产生一个`iterable`； 系统将为之创建一个 *迭代器* ，然后将迭代器所提供的每一项执行一次`suite`，具体次序与迭代器的返回顺序一致
    - 每一项会被依次赋值给目标列表，然后子句体将被执行 - 当所有项被耗尽时（这会在序列为空或迭代器引发`StopIteration`异常时立刻发生），`else`子句的子句体如果存在将会被执行，并终止循环
    - 第一个子句体中的`break`语句在执行时将终止循环且**不**执行`else` 子句体
    - 第一个子句体中的`continue`语句在执行时将跳过子句体中的剩余部分并转往下一项继续执行，或者在没有下一项时转往`else`子句执行
    - `for`循环会对目标列表中的变量进行赋值。 这将覆盖之前对这些变量的所有赋值，包括在`for`循环体中的赋值
    ```
    for i in range(10):
    print(i)
    i = 5             # this will not affect the for-loop
                      # because i will be overwritten with the next
                      # index in the range
    ```
    - 目标列表中的名称（ *循环变量* ）在循环结束时**不会**被删除，但如果序列为空，则它们根本不会被循环所赋值
    - 提示：内置函数[`range`](https://docs.python.org/zh-cn/3.8/library/stdtypes.html#range)会返回一个可迭代的整数序列
- 在遍历同一个集合时修改该集合的代码可能很难获得正确的结果。通常，更直接的做法是 *遍历副本* 或 *创建新集合* 
```
# Strategy:  Iterate over a copy
for user, status in users.copy().items():
    if status == 'inactive':
        del users[user]

# Strategy:  Create a new collection
active_users = {}
for user, status in users.items():
    if status == 'active':
        active_users[user] = status
```

#### 函数参数
    
- 参数传递
    - 函数的 *执行* 会引入一个用于函数局部变量的新 *符号表* 。 更确切地说，函数中所有的变量赋值都将存储在局部符号表中；而变量引用会首先在局部符号表中查找，然后是外层函数的局部符号表，再然后是全局符号表，最后是内置名称的符号表。 因此，全局变量和外层函数的变量不能在函数内部直接赋值（除非是在 `global`语句中定义的全局变量，或者是在`nonlocal`语句中定义的外层函数的变量），尽管它们可以被引用
    - 在函数 *被调用* 时，实参会被引入被调用函数的本地符号表中；因此，实参是通过 *通过对象引用调用* 传递的。如果传递的是可变对象，则调用者将看到被调用者对其做出的任何更改（插入到列表中的元素）。当一个函数调用另外一个函数时，将会为该调用创建一个新的本地符号表
    - 函数定义会将函数名称与函数对象在当前符号表中进行关联。 解释器会将该名称所指向的对象识别为用户自定义函数。 其他名称也可指向同一个函数对象并可被用来访问该函数
- 参数默认值
    - 与`C/C++`不同，带默认值参数 *不一定* 要在无默认值参数之后
    - **重要警告**： 默认值只会执行 *一次* 
    - 这条规则在默认值为可变对象（列表、字典以及大多数类实例）时很重要。比如，下面的函数会存储在后续调用中传递给它的参数
    ```
    def f(a, L=[]):
        L.append(a)
        return L

    print(f(1))  # [1]
    print(f(2))  # [1, 2]
    print(f(3))  # [1, 2, 3]
    ```
    - 不想要在后续调用之间 *共享默认值* ，你可以这样写这个函数
    ```
    def f(a, L=None):
        if L is None:
            L = []
        L.append(a)
        return L
    ```
- 关键字参数
    - 在函数调用中，关键字参数必须跟随在位置参数的 *后面* 
    - 传递的所有关键字参数必须与函数接受的其中一个参数匹配
    - 它们的顺序并不重要。这也包括非可选参数
    - 不能对同一个参数多次赋值
    - 当存在一个形式为`**name`的最后一个形参时，它会接收一个字典
    ```
    def cheeseshop(kind, *arguments, **keywords):
        print("-- Do you have any", kind, "?")
        print("-- I'm sorry, we're all out of", kind)
        for arg in arguments:
            print(arg)
        print("-" * 40)
        for kw in keywords:
            print(kw, ":", keywords[kw])
            
    cheeseshop("Limburger", "It's very runny, sir.",
       "It's really very, VERY runny, sir.",
       shopkeeper="Michael Palin",
       client="John Cleese",
       sketch="Cheese Shop Sketch")
       
    -- Do you have any Limburger ?
    -- I'm sorry, we're all out of Limburger
    It's very runny, sir.
    It's really very, VERY runny, sir.
    ----------------------------------------
    shopkeeper : Michael Palin
    client : John Cleese
    sketch : Cheese Shop Sketch
    ```
- 特殊参数：使用`..., /, ..., *, ...`分隔 *位置* 、 *位置——关键字* 和 *关键字* 三类参数
```
def f(pos1, pos2, /, pos_or_kwd, *, kwd1, kwd2):
  -----------    ----------     ----------
    |             |                  |
    |        Positional or keyword   |
    |                                - Keyword only
     -- Positional only
```
- 可变数量参数：`*args`
```
>>> def concat(*args, sep="/"):
...     return sep.join(args)
...
>>> concat("earth", "mars", "venus")
'earth/mars/venus'
>>> concat("earth", "mars", "venus", sep=".")
'earth.mars.venus'
```
- 解包参数列表
    - 解包列表：`*args`
    ```
    >>> list(range(3, 6))            # normal call with separate arguments
    [3, 4, 5]
    >>> args = [3, 6]
    >>> list(range(*args))            # call with arguments unpacked from a list
    [3, 4, 5]
    ```
    - 解包字典：`**kwds`
    ```
    >>> def tst(a, b, c):
    ...    print(a)
    ...    print(b)
    ...    print(c)
    
    >>> d = {'a': '1', 'b': '2', 'c': '3'}
    >>> tst(*d)
    a
    b
    c
    >>> tst(**d)
    1
    2
    3
    ```

#### 函数标注

- *装饰器* （decorator）
    - *装饰器* 是返回函数的函数
    - 装饰器的常见例子包括`classmethod()`和`staticmethod()`
    - 装饰器语法只是一种语法糖，以下两个函数定义在语义上完全等价
        - 无参数装饰器
        ```
        def f(...):
            ...
        f = staticmethod(f)

        @staticmethod
        def f(...):
            ...
        ```
        - 带参数装饰器
        ```
        def fun(...):
            ...
        fun = wrapper(arg)(fun)

        @wrapper(arg)
        def fun(...):
            ...  
        ```
    - 一个函数定义可以被一个或多个`@decorator`表达式所包装。 当函数被定义时将在包含该函数定义的作用域中对装饰器表达式求值。 求值结果必须是一个可调用对象，它会以该函数对象作为唯一参数被发起调用。其返回值将被绑定到函数名称而非函数对象。 多个装饰器会以嵌套方式被应用。
    ```
    @f1(arg)
    @f2
    def func(): pass

    def func(): pass
    func = f1(arg)(f2(func))
    ```
    - 装饰器常见定义方法：
    ```
    def log(func):
        def wrapper(*args, **kwds):
            print('{}()'.format(func.__name__))
            return func(*args, **kwds)
        return wrapper
    
    @log
    def now():
        print('2015-3-25')
        
    def log(text):
        def decorator(func):
            def wrapper(*args, **kwds):
                print('{} {}()'.format(text, func.__name__))
                return func(*args, **kwds)
            return wrapper
        return decorator
        
    @log('execute')
    def now():
        print('2015-3-25')
    ```
        
#### `lambda`表达式

- 语法
```
lambda <parameter_list>: expression
```
- `<parameter_list>`为 *可选* 的，整个定义等价于
```
def <lambda>(parameters):
    return expression
```
- `lambda`创建的函数**不能**包含 *语句* 或 *标注* 
- 典型用例：作为谓词传递
```
>>> pairs = [(1, 'one'), (2, 'two'), (3, 'three'), (4, 'four')]
>>> pairs.sort(key=lambda pair: pair[1])
>>> pairs
[(4, 'four'), (1, 'one'), (3, 'three'), (2, 'two')]
```

### 🌱 数据结构

#### 列表的更多特性

- 特性列表
    - `list.append(x)`
        - 在列表的末尾添加一个元素
        - 相当于`a[len(a):] = [x]`
    - `list.extend(iterable)`
        - 使用可迭代对象中的所有元素来扩展列表
        - 相当于`a[len(a):] = iterable`
    - `list.insert(i, x)`
        - 在给定的位置插入一个元素
        - 第一个参数是要插入的元素的索引，所以
            - `a.insert(0, x)`插入列表头部
            - `a.insert(len(a), x)`等同于`a.append(x)`
    - `list.remove(x)`
        - 移除列表中 *第一个* 值为`x`的元素
        - 如果没有这样的元素，则抛出`ValueError`
    - `list.pop([i])`
        - 删除列表中给定位置的元素并返回它
        - `i`为 *可选参数* 。如果没有给定位置，`a.pop()`将会删除并返回列表中的 *最后一个* 元素
    - `list.clear()`
        - 移除列表中的所有元素。等价于`del a[:]`
    - `list.index(x[, start[, end]])`
        - 返回列表中 *第一个* 值为`x` 的元素的从零开始的索引
        - 如果没有这样的元素将会抛出`ValueError`
        - *可选参数* `start`和`end`是切片符号，用于将搜索限制为列表的特定子序列。返回的索引是相对于整个序列的开始计算的，而不是`start`参数
    - `list.count(x)`
        - 返回元素`x`在列表中出现的次数
    - `list.sort(key=None, reverse=False)`
        - 对列表中的元素进行排序
    - `list.reverse()`
        - 翻转列表中的元素
    - `list.copy()`
        - 返回列表的一个 *浅* 拷贝，等价于`a[:]`
- 列表作为 *栈* 使用
    - 调用`list.append(x)`和`list.pop()`
- **不推荐** 将列表作为 *队列* 使用
    - `append`很快，但`insert(0, x)`需移动整个列表，很慢
    - 使用[`collections.deque`](https://docs.python.org/zh-cn/3.8/library/collections.html#collections.deque)
    ```
    >>> from collections import deque
    >>> queue = deque(["Eric", "John", "Michael"])
    >>> queue.append("Terry")           # Terry arrives
    >>> queue.append("Graham")          # Graham arrives
    >>> queue.popleft()                 # The first to arrive now leaves
    'Eric'
    >>> queue.popleft()                 # The second to arrive now leaves
    'John'
    >>> queue                           # Remaining queue in order of arrival
    deque(['Michael', 'Terry', 'Graham'])
    ```
- *列表推导式* 
    - 列表推导式的结构是由一对方括号所包含的以下内容：一个表达式，后面跟一个`for`子句，然后是零个或多个`for`或`if`子句。 其结果将是一个新列表，由对表达式依据后面的`for`和`if`子句的内容进行求值计算而得出
    - 如果表达式是 *元组* ，则必须加上 *括号* 
    - 例如，如下两段代码等价
    ```
    >>> [(x, y) for x in [1, 2, 3] for y in [3, 1, 4] if x != y]
    [(1, 3), (1, 4), (2, 3), (2, 1), (2, 4), (3, 1), (3, 4)]
    
    >>> combs = []
    >>> for x in [1, 2, 3]:
    ...     for y in [3, 1, 4]:
    ...         if x != y:
    ...             combs.append((x, y))
    ...
    >>> combs
    [(1, 3), (1, 4), (2, 3), (2, 1), (2, 4), (3, 1), (3, 4)]
    ```
    - 列表推导式可以使用复杂的表达式和嵌套函数
    ```
    >>> from math import pi
    >>> [str(round(pi, i)) for i in range(1, 6)]
    ['3.1', '3.14', '3.142', '3.1416', '3.14159']
    ```
- 嵌套的列表推导式
    - 示例：转置矩阵
    ```
    >>> matrix = [
    ...     [1,  2,  3,  4],
    ...     [5,  6,  7,  8],
    ...     [9, 10, 11, 12],
    ... ]
    >>> [[row[i] for row in matrix] for i in range(4)]
    [[1, 5, 9], [2, 6, 10], [3, 7, 11], [4, 8, 12]] 
    
    # equals to
    >>> transposed = []
    >>> for i in range(4):
    ...     # the following 3 lines implement the nested listcomp
    ...     transposed_row = []
    ...     for row in matrix:
    ...         transposed_row.append(row[i])
    ...     transposed.append(transposed_row)
    ...
    >>> transposed
    [[1, 5, 9], [2, 6, 10], [3, 7, 11], [4, 8, 12]]
    ```
    - 上例使用[`zip`](https://docs.python.org/zh-cn/3.8/library/functions.html#zip)会更方便
    ```
    >>> list(zip(*matrix))
    [(1, 5, 9), (2, 6, 10), (3, 7, 11), (4, 8, 12)]
    ```

#### `del`语句

- 语法：`del <target_list>`
- 删除是递归定义的，与赋值的定义方式非常类似。目标列表`<target_list>`的删除将从左至右递归地删除每一个目标
- 名称的删除将从局部或全局命名空间中移除该名称的绑定，具体作用域的确定是看该名称是否有在同一代码块的`global`语句中出现。 如果该名称未被绑定，将会引发`NameError`
- *属性引用* 、 *抽取* 和 *切片* 的删除会被传递给相应的原型对象；删除一个切片基本等价于赋值为一个右侧类型的空切片（但即便这一点也是由切片对象决定的）
- 示例
```
>>> a = [-1, 1, 66.25, 333, 333, 1234.5]
>>> del a[0]
>>> a
[1, 66.25, 333, 333, 1234.5]
>>> del a[2:4]
>>> a
[1, 66.25, 1234.5]
>>> del a[:]
>>> a
[]
>>> del a
>>> a
NameError: name 'a' is not defined
```

#### 元组和序列

- 我们看到列表和字符串有很多共同特性，例如 *索引* 和 *切片* 操作。他们是 *序列* 类型（`list`、`tuple`和`range`）中的两种
- 输入元组时圆括号可有可无，不过经常会是必须的（如果这个元组是一个更大的表达式的一部分）
- 给元组中的一个单独的元素赋值是**不允许**的，当然你可以创建包含可变对象的元组，例如列表
- 元组和列表的区别
    - 元组是`immutable`，其序列通常包含 *不同种类* 的元素，并且通过 *解包* 或者 *索引* 来访问（如果是`namedtuples` 的话甚至还可以通过 *属性* 访问）。
    - 列表是`mutable`，并且列表中的元素一般是 *同种类型* 的，并且通过 *迭代* 访问
- 空元组和单一元素元组
    - 创建空元组：空的括号
    - 创建单元素元组：一个元素加一个逗号
        - 丑陋，但是有效
    ```
    >>> empty = ()
    >>> singleton = 'hello',    # <-- note trailing comma
    >>> len(empty)
    0
    >>> len(singleton)
    1
    >>> singleton
    ('hello',)
    ```
- 元组打包和解包
```
>>> t = 12345, 54321, 'hello!'
>>> t
(12345, 54321, 'hello!')
>>> x, y, z = t
>>> x, y, z
12345 54321 hello!
```

#### 集合

- 集合是由 *不重复* 元素组成的无序的集。它的基本用法包括成员检测和消除重复元素。集合对象也支持像 *联合* ， *交集* ， *差集* ， *对称差分* 等数学运算
- 创建集合
    - *花括号* 或[`set`](https://docs.python.org/zh-cn/3.8/library/stdtypes.html#set) 函数可以用来创建集合
    - 注意：要创建一个空集合你只能用`set()`而**不能**用`{}`，因为后者是创建一个 *空字典* 
- 集合推导式
    - 类似于列表推导式
    ```
    >>> a = {x for x in 'abracadabra' if x not in 'abc'}
    >>> a
    {'r', 'd'}
    ```

#### 字典

- 字典是 *键值对* 的集合
    - *映射* 类型：以 *关键字* 为索引，关键字可以是任意 *不可变* 类型，通常是字符串或数字
        - 如果一个元组只包含字符串、数字或元组，那么这个元组也可以用作关键字
        - 但如果元组直接或间接地包含了可变对象，那么它就**不能**用作关键字
        - 列表**不能**用作关键字，因为列表可以通过索引、切片或`append()`和 `extend()`之类的方法来改变
- 创建字典
    - 一对 *花括号* 可以创建一个空字典：`{}` 
    - 另一种初始化字典的方式是在一对 *花括号* 里放置一些以 *逗号* 分隔的键值对，而这也是字典输出的方式
    - [`dict`构造函数](https://docs.python.org/zh-cn/3.8/library/stdtypes.html#dict)可以直接从键值对序列里创建字典
    - 字典推导式
    ```
    >>> {x: x ** 2 for x in (2, 4, 6)}
    {2: 4, 4: 16, 6: 36}
    ```
- 在一个字典中，键必须是唯一的
    - 如果你使用了一个已经存在的关键字来存储值，那么之前与这个关键字关联的值就会被遗忘
    - 用一个不存在的键来取值则会报错
- 可以用`del`来删除一个 *键值对* 
- 对一个字典执行`list(d)` 将返回包含该字典中所有键的列表，按插入次序排列（如需其他排序，则要使用 `sorted(d)`）。要检查字典中是否存在一个特定键，可使用[`in`](https://docs.python.org/zh-cn/3.8/reference/expressions.html#in)关键字

#### 循环的技巧

- 当在字典中循环时，用`items()`方法可将关键字和对应的值同时取出
```
>>> knights = {'gallahad': 'the pure', 'robin': 'the brave'}
>>> for k, v in knights.items():
...     print(k, v)
...
gallahad the pure
robin the brave
```
- 当在序列中循环时，用`enumerate()`函数可以将索引位置和其对应的值同时取出
```
>>> for i, v in enumerate(['tic', 'tac', 'toe']):
...     print(i, v)
...
0 tic
1 tac
2 toe
```
- 当同时在两个或更多序列中循环时，可以用`zip()`函数将其内元素一一匹配
```
>>> questions = ['name', 'quest', 'favorite color']
>>> answers = ['lancelot', 'the holy grail', 'blue']
>>> for q, a in zip(questions, answers):
...     print('What is your {0}?  It is {1}.'.format(q, a))
...
What is your name?  It is lancelot.
What is your quest?  It is the holy grail.
What is your favorite color?  It is blue.
```
- 如果要逆向循环一个序列，可以先正向定位序列，然后调用`reversed()`函数
```
>>> for i in reversed(range(1, 10, 2)):
...     print(i)
...
9
7
5
3
1
```
- 如果要按某个指定顺序循环一个序列，可以用`sorted()`函数，它可以在不改动原序列的基础上返回一个新的排好序的序列
```
>>> basket = ['apple', 'orange', 'apple', 'pear', 'orange', 'banana']
>>> for f in sorted(set(basket)):
...     print(f)
...
apple
banana
orange
pear
```
- 有时可能会想在循环时修改列表内容，一般来说改为创建一个新列表是比较简单且安全的
```
>>> import math
>>> raw_data = [56.2, float('NaN'), 51.7, 55.3, 52.5, float('NaN'), 47.8]
>>> filtered_data = []
>>> for value in raw_data:
...     if not math.isnan(value):
...         filtered_data.append(value)
...
>>> filtered_data
[56.2, 51.7, 55.3, 52.5, 47.8]
```

#### 深入条件控制

- `while`和`if`条件句中可以使用任意操作，而不仅仅是比较操作
- 比较操作符
    - 操作符`in`和`not in`校验一个值是否在（或不在）一个序列里。
    - 操作符`is`和`is not`比较两个对象是不是同一个对象，这只对像列表这样的可变对象比较重要
    - 所有的比较操作符都有相同的优先级，且这个优先级比数值运算符低
- 比较操作可以传递。例如`a < b == c`会校验是否`a`小于`b`并且`b` 等于`c`
- 比较操作可以通过布尔运算符`and`和`or`来组合，并且比较操作（或其他任何布尔运算）的结果都可以用`not`来取反。这些操作符的优先级低于比较操作符；在它们之中，`not`优先级最高，`or`优先级最低，因此`A and not B or C`等价于`(A and (not B)) or C`。和之前一样，你也可以在这种式子里使用圆括号
- 布尔运算符`and`和`or`也被称为 *短路运算符* ：它们的参数从左至右解析，一旦可以确定结果解析就会停止。例如，如果`A`和`C`为真而`B`为假，那么`A and B and C`不会解析`C`。当用作普通值而非布尔值时，短路操作符的返回值通常是最后一个变量
- 也可以把比较操作或者逻辑表达式的结果赋值给一个变量，例如
```
>>> string1, string2, string3 = '', 'Trondheim', 'Hammer Dance'
>>> non_null = string1 or string2 or string3
>>> non_null
'Trondheim'
```
- 注意`python`与`C`不同，赋值操作**不能**发生在表达式内部。`C`程序员可能会对此抱怨，但它避免了一类`C`程序中常见的错误：想在表达式中写`==`时却写成了`=`

### 🌱 `I/O`

- `format`的`{}`可以写成`{:fmt}`，`fmt`为`C`语言`printf`风格的格式化字符串（没有开头的`%`）

### 🌱 类

#### 名称和对象

- 对象具有个性，多个名称（在多个作用域内）可以绑定到同一个对象。这在其他语言中称为别名
- `python`中的对象就像是`C`中的指针，传递很快，且对参数的更改对函数外部可见

#### 作用域和命名空间

- 把任何跟在一个点号之后的名称都称为 *属性* 
    - 在表达式`z.real`中，`real`是对象`z`的一个属性
    - 对模块中名称的引用属于 *属性引用* ：在表达式`modname.funcname`中，`modname`是一个模块对象而`funcname`是它的一个属性
        - 在此情况下在模块的属性和模块中定义的全局名称之间正好存在一个直观的映射：它们共享相同的命名空间
- 属性可以是 *只读* 或者 *可写* 的。如果为后者，那么对属性的赋值是可行的。模块属性是可以写，你可以写出 `modname.the_answer = 42`。可写的属性同样可以用`del`语句删除。例如， `del modname.the_answer`将会从名为`modname`的对象中移除`the_answer`属性
- 在不同时刻创建的命名空间拥有不同的 *生存期* 
    - 包含 *内置名称* 的模块（命名空间）`builtin`是在`python` 解释器启动时创建的，永远不会被删除
    - 模块的全局命名空间在模块定义被读入时创建；通常，模块命名空间也会持续到解释器退出
    - 被解释器的顶层调用执行的语句，从一个脚本文件读取或交互式地读取，被认为是`__main__`模块调用的一部分，因此它们拥有自己的全局命名空间
    - 一个 *函数的本地命名空间* 在这个函数被调用时创建，并在函数返回或抛出一个不在函数内部处理的错误时被删除
    - 当然，每次递归调用都会有它自己的本地命名空间
- 作用域
    - 一个 *作用域* 是一个命名空间可直接访问的`python`程序的文本区域。 这里的 *可直接访问* 意味着对名称的非限定引用会尝试在命名空间中查找名称
    - 作用域被静态确定，但被动态使用。 在程序运行的任何时间，至少有三个命名空间可被直接访问的嵌套作用域
        1. 最先搜索的最内部作用域包含的局部名称
        2. 从最近的封闭作用域开始搜索的任何封闭函数的作用域包含非局部名称，也包括非全局名称
        3. 倒数第二个作用域包含当前模块的全局名称
        4. 最外面的作用域（最后搜索）是包含内置名称的命名空间
    - 如果一个名称被声明为 *全局变量* ，则所有引用和赋值将直接指向包含该模块的全局名称的中间作用域。 要重新绑定在最内层作用域以外找到的变量，可以使用`nonlocal`语句声明为非本地变量 如果没有被声明为非本地变量，这些变量将是只读的（尝试写入这样的变量只会在最内层作用域中创建一个 *新的* 局部变量，而同名的外部变量保持不变）
    - 通常，当前局部作为域将（按字面文本）引用当前函数的局部名称。 在函数以外，局部作用域将引用与全局作用域相一致的命名空间：模块的命名空间。类定义将在局部命名空间内再放置另一个命名空间。
- `global`和`nonlocal`语句
    - 如果不存在生效的`global`或`nonlocal`语句，则对名称的赋值 *总是* 会进入 *最内层作用域* 
        - 赋值不会复制数据，它们只是将名称绑定到对象
        - 删除也是如此：语句`del x`会从局部作用域所引用的命名空间中移除对`x`的绑定
    - 事实上，所有引入新名称的操作都是使用局部作用域：特别地，`import` 语句和函数定义会在局部作用域中绑定模块或函数名称。
    - `global` 语句可被用来表明特定变量生存于全局作用域并且应当在其中被重新绑定
    - `nonlocal` 语句表明特定变量生存于外层作用域中并且应当在其中被重新绑定

#### 示例

```
def scope_test():
    def do_local():
        spam = "local spam"

    def do_nonlocal():
        nonlocal spam
        spam = "nonlocal spam"

    def do_global():
        global spam
        spam = "global spam"

    spam = "test spam"
    do_local()
    print("After local assignment:", spam)     # After local assignment: test spam
    do_nonlocal()
    print("After nonlocal assignment:", spam)  # After nonlocal assignment: nonlocal spam
    do_global()
    print("After global assignment:", spam)    # After global assignment: nonlocal spam

scope_test()
print("In global scope:", spam)                # In global scope: global spam
```

#### 类变量和实例变量

- 一般来说，实例变量用于每个实例的唯一数据，而类变量用于类的所有实例共享的属性和方法
```
class Dog:

    kind = 'canine'         # class variable shared by all instances

    def __init__(self, name):
        self.name = name    # instance variable unique to each instance

>>> d = Dog('Fido')
>>> e = Dog('Buddy')
>>> d.kind                  # shared by all dogs
'canine'
>>> e.kind                  # shared by all dogs
'canine'
>>> d.name                  # unique to d
'Fido'
>>> e.name                  # unique to e
'Buddy'
```


### 🌱 `import`

```
"""
Illustration of good import statement styling.
Note that the imports come after the docstring.
"""

# Standard library imports
import datetime
import os

# Third party imports
from flask import Flask
from flask_restful import Api
from flask_sqlalchemy import SQLAlchemy

# Local application imports
from local_module import local_class
from local_package import local_function
```




### 🌱 [标准库](https://docs.python.org/zh-cn/3.7/library/index.html)

- 比较牛B，就不像隔壁`C++`那样抄了，实在抄不完






### 🌱 [类型标注](https://docs.python.org/zh-cn/3.7/library/typing.html)

- 概述
    - `Python runtime`不强迫执行函数以及变量类型标注。类型标注是为了方便第三方工具，例如`type checker`、`IDE`以及`linters`等。
    - 最基本的类型标注由[`Any`](https://docs.python.org/zh-cn/3.7/library/typing.html#typing.Any)、[`Union`](https://docs.python.org/zh-cn/3.7/library/typing.html#typing.Union)、[`Tuple`](https://docs.python.org/zh-cn/3.7/library/typing.html#typing.Tuple)、[`Callable`](https://docs.python.org/zh-cn/3.7/library/typing.html#typing.Callable)、[`TypeVar`](https://docs.python.org/zh-cn/3.7/library/typing.html#typing.TypeVar)和[`Generic`](https://docs.python.org/zh-cn/3.7/library/typing.html#typing.Any)类型组成。
    - 例如，可以指定如下函数接受并返回一个字符串（`str`的子类型也可以）
    ```
    def greeting(name: str) -> str: 
        return 'Hello' + name
    ```
- *类型别名* 
    - *类型别名* 通过将类型分配给别名来定义。
    - 在这个例子中，`Vector`和`List[float]`将被视为可互换的同义词：
    ```
    from typing import List
    Vector = List[float]

    def scale(scalar: float, vector: Vector) -> Vector:
        return [scalar * num for num in vector]

    # typechecks; a list of floats qualifies as a Vector.
    new_vector = scale(2.0, [1.0, -4.2, 5.4])
    ```
    - 类型别名可用于简化复杂类型签名。例如：
    ```
    from typing import Dict, Tuple, Sequence

    ConnectionOptions = Dict[str, str]
    Address = Tuple[str, int]
    Server = Tuple[Address, ConnectionOptions]

    def broadcast_message(message: str, servers: Sequence[Server]) -> None:
        ...

    # The static type checker will treat the previous type signature as
    # being exactly equivalent to this one.
    def broadcast_message
    (
        message: str,
        servers: Sequence[Tuple[Tuple[str, int], Dict[str, str]]]
    ) -> None:
        ...
    ```
    - 请注意，`None`作为类型提示是一种特殊情况：应当写作`type(None)`
- `NewType`
    - 使用[`NewType`](https://docs.python.org/zh-cn/3.7/library/typing.html#typing.NewType)辅助函数创建不同的类型
    ```
    from typing import NewType

    UserId = NewType('UserId', int)
    some_id = UserId(524313)
    ```
    - 静态类型检查器会将新类型视为其 *原始类型的子类* 。这对于帮助捕捉逻辑错误非常有用：
    ```
    def get_user_name(user_id: UserId) -> str:
        ...

    # typechecks
    user_a = get_user_name(UserId(42351))

    # does not typecheck; an int is not a UserId
    user_b = get_user_name(-1)
    ```
    - 您仍然可以对`UserId`类型的变量执行所有的`int`支持的操作，但结果将始终为`int`类型。这可以让你在需要`int` 的地方传入`UserId`，但会**阻止**你以无效的方式无意中创建`UserId`：
    ```
    # 'output' is of type 'int', not 'UserId'
    output = UserId(23413) + UserId(54341)
    ```
    - 请注意，这些检查仅通过静态类型检查程序强制执行。在运行时，`Derived = NewType('Derived'，Base)`是创建了一个名为`Derived`的函数，该函数立即返回您传递它的任何参数。这意味着表达式 `Derived(some_value)` **不会**创建一个新的类或引入任何超出常规函数调用的开销。更确切地说，表达式 `some_value is Derived(some_value)`在运行时总是为`True`。这也意味着**无法**创建`Derived`的派生类型，因为它是运行时的标识函数，而不是实际的类型：
    ```
    from typing import NewType

    UserId = NewType('UserId', int)

    # Fails at runtime and does not typecheck
    class AdminUserId(UserId): pass
    ```
    - 但是，可以用`NewType`函数套娃，并且`ProUserId`的类型检查将按预期工作：
    ```
    from typing import NewType

    UserId = NewType('UserId', int)

    ProUserId = NewType('ProUserId', UserId)
    ```
    - **注解**
        - 回想一下，使用类型别名声明两种类型彼此 *等效* 。`Alias = Original`将使静态类型检查对待所有情况下`Alias`完全等同于`Original`。当您想简化复杂类型签名时，这很有用。
        - 相反，`NewType`声明一种类型是另一种类型的 *子类型* 。`Derived = NewType('Derived', Original)`将使静态类型检查器将 `Derived`当作`Original`的 *子类* ，这意味着`Original`类型的值**不能**用于`Derived`类型的值需要的地方。当您想以最小的运行时间成本防止逻辑错误时，这非常有用。
- `Callable`
    - 期望特定签名的回调函数的框架可以将类型标注为`Callable[[ArgType, Arg2Type], ReturnType]`，例如：
    ```
    from typing import Callable

    def feeder(get_next_item: Callable[[], str]) -> None:
        # Body

    def async_query
    (
        on_success: Callable[[int], None],
        on_error: Callable[[int, Exception], None]
    ) -> None:
        # Body
    ```
    - 通过用 *文字省略号* `...` 替换类型提示中的参数列表：`Callable[...，ReturnType]`，可以声明可调用的返回类型，而无需指定调用签名。
- 泛型（Generic）
    - 由于无法以通用方式静态推断有关保存在容器中的对象的类型信息，因此抽象基类已扩展为支持订阅以表示容器元素的预期类型。
    ```
    from typing import Mapping, Sequence

    def notify_by_email
    (
        employees: Sequence[Employee],
        overrides: Mapping[str, str]
    ) -> None: 
        ...
    ```
    - 泛型可以通过使用`typing`模块中名为`TypeVar`的新工厂进行参数化。
    ```
    from typing import Sequence, TypeVar

    T = TypeVar('T')      # Declare type variable

    def first(l: Sequence[T]) -> T:   # Generic function
        return l[0]
    ```
- `Any`类型
    - `Any`是一种特殊的类型。静态类型检查器将所有类型视为与`Any`兼容，反之亦然，`Any`也与所有类型相兼容。这意味着可对类型为`Any`的值执行任何操作或方法调用，并将其赋值给任何变量：
    ```
    from typing import Any

    a = None    # type: Any
    a = []      # OK
    a = 2       # OK

    s = ''      # type: str
    s = a       # OK

    def foo(item: Any) -> int:
        # Typechecks; 'item' could be any type,
        # and that type might have a 'bar' method
        item.bar()
        ...
    ```
    - 需要注意的是，将`Any`类型的值赋值给另一个更具体的类型时，`Python`**不会**执行类型检查。例如，当把`a`赋值给`s`时，即使`s`被声明为`str`类型，在运行时接收到的是`int`值，静态类型检查器也**不会**报错。
    - 此外，所有 *返回值无类型* 或 *形参无类型* 的 *函数* 将 *隐式地默认使用* `Any`类型：
    ```
    def legacy_parser(text):
        ...
        return data

    # A static type checker will treat the above
    # as having the same signature as:
    def legacy_parser(text: Any) -> Any:
        ...
        return data
    ```
    - 当需要混用动态类型和静态类型的代码时，上述行为可以让`Any`被用作 应急出口。

    - `Any`和`object`的行为对比
        - 与`Any`相似，所有的类型都是`object`的子类型。然而不同于`Any`，反之并不成立：`object`**不是**其他所有类型的子类型。这意味着当一个值的类型是`object`的时候，类型检查器会拒绝对它的几乎所有的操作。把它赋值给一个指定了类型的变量（或者当作返回值）是一个类型错误。比如说：
        ```
        def hash_a(item: object) -> int:
            # Fails; an object does not have a 'magic' method.
            item.magic()
            ...

        def hash_b(item: Any) -> int:
            # Typechecks
            item.magic()
            ...

        # Typechecks, since ints and strs are subclasses of object
        hash_a(42)
        hash_a("foo")

        # Typechecks, since Any is compatible with all types
        hash_b(42)
        hash_b("foo")
        ```
        - 使用`object`示意一个值可以类型安全地兼容任何类型。使用`Any`示意一个值地类型是动态定义的。








