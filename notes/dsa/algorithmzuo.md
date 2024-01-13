# algorithmzuo



## 006 Binary Search

```c++
// 二分查找：在有序数组的区间 a[lo, hi) 内查找元素 num
int binSearch(int a, int lo, int hi, int num)
{
    while (lo < hi)
    {
        int mi = lo + ((hi - lo) >> 1);

        if      (num < A[mi]) hi = mi;
        else if (A[mi] < num) lo = mi + 1;
        else                  return mi;
    }

    return -1;
}

// 有多个命中元素时，总能保证返回秩最大者；查找失败时，能够返回失败的位置
int binSearch(int a, int lo, int hi, int num)
{
    while (lo < hi)
    {
        int mi = lo + ((hi - lo) >> 1);
        (num < A[mi]) ? hi = mi : lo = mi + 1;
    }

    return --lo;
}

// 有序数组中找 >= num 的最左位置
int lowerBound(int a, int lo, int hi, int num)
{
    int ans = -1;
    
    while (lo < hi)
    {
        int mi = lo + ((hi - lo) >> 1);

        if (num <= a[mi])
        {
            ans = mi;
            hi = mi;
        }
        else
        {
            lo = mi + 1;
        }
    }

    return ans;
}

// 有序数组中找 <= num 的最右位置
int upperBound(int a, int lo, int hi, int num)
{
    int ans = -1;
    
    while (lo < hi)
    {
        int mi = lo + ((hi - lo) >> 1);

        if (a[mi] <= num)
        {
            ans = mi;
            lo = mi + 1;
        }
        else
        {
            hi = mi;
        }
    }

    return ans;
}

// LC 162 https://leetcode.com/problems/find-peak-element/
// 峰值元素是指其值严格大于左右相邻值的元素
// 给你一个整数数组 nums，已知任何两个相邻的值都不相等
// 找到峰值元素并返回其索引
// 数组可能包含多个峰值，在这种情况下，返回 任何一个峰值 所在位置即可。
// 你可以假设 nums[-1] = nums[n] = 无穷小
// 你必须实现时间复杂度为 O(log n) 的算法来解决此问题。
int findPeakElement(std::vector<int> & nums) 
{
    if (nums.size() == 1 || nums[1] < nums.front())
    {
        return 0;
    }

    if (*(nums.end() - 2) < nums.back()) 
    {
        return nums.size() - 1;
    }

    int lo = 1, hi = nums.size() - 1, ans = -1;

    // If the mid element happens to be lying in a local falling slope, 
    // it means that the peak will always lie towards the left of this element.
    while (lo < hi)
    {
        int mi = lo + ((hi - lo) >> 1);

        if (nums[mi] < nums[mi - 1])
        {
            hi = mi;
        }
        else if (nums[mi] < nums[mi + 1])
        {
            lo = mi + 1;
        }
        else
        {
            ans = mi;
            break;
        }
    }

    return ans;
}
```

## 016 Circular Deque

```c++
/// LC 641. Design Circular Deque
/// https://leetcode.com/problems/design-circular-deque/
class MyCircularDeque 
{
public:
    MyCircularDeque(int k) : l(0), r(0), size(0), limit(k), deque(k)
    {

    }

    bool insertFront(int value) 
    {
        if (isFull()) 
        {
            return false;
        } 

        if (isEmpty()) 
        {
            l = r = 0;
            deque[0] = value;
        } 
        else 
        {
            l = (l == 0) ? (limit - 1) : (l - 1);
            deque[l] = value;
        }

        size++;
        return true;
    }

    bool insertLast(int value) 
    {
        if (isFull()) 
        {
            return false;
        } 

        if (isEmpty()) 
        {
            l = r = 0;
            deque[0] = value;
        } 
        else 
        {
            r = (r == limit - 1) ? 0 : (r + 1);
            deque[r] = value;
        }

        size++;
        return true;
    }

    bool deleteFront() 
    {
        if (isEmpty()) 
        {
            return false;
        } 

        l = (l == limit - 1) ? 0 : (l + 1);
        size--;
        return true;
    }

    bool deleteLast() 
    {
        if (isEmpty()) 
        {
            return false;
        } 

        r = (r == 0) ? (limit - 1) : (r - 1);
        size--;
        return true;
    }

    int getFront() 
    {
        return isEmpty() ? -1 : deque[l];
    }

    int getRear() 
    {
        return isEmpty() ? -1 : deque[r];
    }

    bool isEmpty() 
    {
        return size == 0;
    }

    bool isFull() 
    {
        return size == limit;
    }

private:
    std::vector<int> deque;
    int l;
    int r;
    int size;
    int limit;
};
```























