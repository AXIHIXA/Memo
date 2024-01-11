# Bit Operations

## Bitwise AND

- `x & (x - 1)`: Unset lowest 1 bit;
  - Usage: Bin count, etc. 
- `x & (-x)` or `x & (~x + 1)`: Extract lowest 1 bit (unset all other bits);
- If `n` is power of 2, then `x % n` equals `x & (n - 1)`. 

## Bitwise XOR

- Carry-less addition. 
  - Communicative; 
  - Associative. 
- XOR zero equals self: `x ^ 0 == x`;
- XOR self equals zero: `x ^ x == 0`;
- `swap`: `a ^= b, b ^= a, a ^= b;` 
  - Note that this expression is wrong when `&a == &b`!
- Missing Number: `a[0:n]`, ranging from 0 to n, missing one number. 
  - `n ^ (i ^ a[i] for i in range(n))`
- One number odd occurance, all other numbers even occurance, find this number: 
  - XOR all elements in array and the result is this odd-occurance number;
- Two numbers odd occurence, all other numbers even occurence, find these two numbers:
  - XOR all elements in array, extract lowest 1 bit in the result;
  - The elements in this array could be divided in two groups (checking this extracted bit);
  - Each group contains one of the two odd-occurance numbers. 
- One number occurs less than m times, all other numbers occur exactly m times, find this number:
  - Calculate the number of 1-bits for bit 0 to 31 for all elements in this array;
  - Extract bits whose number of 1-bits is not multiple of m;
  - The result is the extracted number. 

## Zuo's Sample Problems

```c++
// Brian Kernighan算法
// 提取出二进制里最右侧的1
// 判断一个整数是不是2的幂
// 测试链接 : https://leetcode.cn/problems/power-of-two/
bool isPowerOfTwo(int n) 
{
    return n > 0 && n == (n & -n);
}

// 判断一个整数是不是3的幂
// 测试链接 : https://leetcode.cn/problems/power-of-three/
// 如果一个数字是3的某次幂，那么这个数一定只含有3这个质数因子
// 1162261467是int型范围内，最大的3的幂，它是3的19次方
// 这个1162261467只含有3这个质数因子，如果n也是只含有3这个质数因子，那么
// 1162261467 % n == 0
// 反之如果1162261467 % n != 0 说明n一定含有其他因子
bool isPowerOfThree(int n) 
{
    return n > 0 && 1162261467 % n == 0;
}

// 已知n是非负数
// 返回大于等于n的最小的2某次方
// 如果int范围内不存在这样的数，返回整数最小值
int near2power(int n) 
{
    if (n <= 0) 
    {
        return 1;
    }

    n--;
    n |= n >>> 1;
    n |= n >>> 2;
    n |= n >>> 4;
    n |= n >>> 8;
    n |= n >>> 16;
    return n + 1;
}

// 给你两个整数 left 和 right ，表示区间 [left, right]
// 返回此区间内所有数字 & 的结果
// 包含 left 、right 端点
// 测试链接 : https://leetcode.cn/problems/bitwise-and-of-numbers-range/
int rangeBitwiseAnd(int left, int right) 
{
    while (left < right) 
    {
        right -= right & -right;
    }

    return right;
}


// 逆序二进制的状态
// 测试链接 : https://leetcode.cn/problems/reverse-bits/
int reverseBits(int n) 
{
    n = ((n & 0xaaaaaaaa) >>> 1) | ((n & 0x55555555) << 1);
    n = ((n & 0xcccccccc) >>> 2) | ((n & 0x33333333) << 2);
    n = ((n & 0xf0f0f0f0) >>> 4) | ((n & 0x0f0f0f0f) << 4);
    n = ((n & 0xff00ff00) >>> 8) | ((n & 0x00ff00ff) << 8);
    n = (n >>> 16) | (n << 16);
    return n;
}

// 返回n的二进制中有几个1
int binCount(unsigned int n)
{
    n = (n & 0x55555555U) + ((n & 0xaaaaaaaaU) >> 1U);
    n = (n & 0x33333333U) + ((n & 0xccccccccU) >> 2U);
    n = (n & 0x0f0f0f0fU) + ((n & 0xf0f0f0f0U) >> 4U);
    n = (n & 0xff00ffU)   + ((n & 0xff00ff00U) >> 8U);
    n = (n & 0xffffU)     + ((n & 0xffff0000U) >> 16U);
    return n;
}
```
