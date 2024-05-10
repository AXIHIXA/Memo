# algorithmzuo


## 000 Implementation Notes

- [Implementation Notes - STL](./implementation_notes_stl.md)
- [Implementation Notes - Coding/Algorithm](./implementation_notes_lc.md)



## 006 Binary Search

```c++
// 二分查找：在有序数组的区间 a[lo, hi) 内查找元素 e
int binSearch(int a, int lo, int hi, int e)
{
    while (lo < hi)
    {
        int mi = lo + ((hi - lo) >> 1);

        if      (e < a[mi]) hi = mi;
        else if (a[mi] < e) lo = mi + 1;
        else                return mi;
    }

    return -1;
}

// Analogous to std::lower_bound. 
// Locates in a[lo, hi) for 1st a[i] s.t. e <= a[i]. 
// See libstdc++: 
// https://github.com/gcc-mirror/gcc/blob/d9375e490072d1aae73a93949aa158fcd2a27018/libstdc%2B%2B-v3/include/bits/stl_algobase.h#L1023
int lowerBound(int * a, int lo, int hi, int e)
{
    while (lo < hi)
    {
        int mi = lo + ((hi - lo) >> 1);
        if (a[mi] < e) lo = mi + 1;
        else hi = mi;
    }

    return lo;
}

// Analogous to std::upper_bound. 
// Locates in a[lo, hi) for 1st a[i] s.t. e < a[i]. 
// See libstdc++:
// https://github.com/gcc-mirror/gcc/blob/d9375e490072d1aae73a93949aa158fcd2a27018/libstdc%2B%2B-v3/include/bits/stl_algo.h#L2028
int upperBound(int * a, int lo, int hi, int e)
{
    while (lo < hi)
    {
        int mi = lo + ((hi - lo) >> 1);
        if (e < a[mi]) hi = mi;
        else lo = mi + 1;
    }

    return lo;
}

// Equal range: 
// ... < < < < = = = = = < < < < ...
//             |         |
//            ll         rr
//    lowerBound         upperBound
```
- [LC 162 Find Peak Element](https://leetcode.com/problems/find-peak-element/)
  - A peak element is an element that is strictly greater than its neighbors.
  - Binary search. Go left/right corresponding on slope of prev curr next. 



## 016 Circular Deque

- [LC 641. Design Circular Deque](https://leetcode.com/problems/design-circular-deque/)



## 018 Binary Tree Traversal Iteration

```c++
#include <iostream>
#include <stack>
#include <vector>

struct Node
{
    Node() = default;
    Node(int v) : val(v) {}
    Node(int v, Node * l, Node * r) : val(v), left(l), right(r) {}

    int val {0};
    Node * left {nullptr};
    Node * right {nullptr};
};

// Root Left Right. 
void preorderTraverse(Node * head)
{
    if (!head) return;

    std::stack<Node *> st;
    st.emplace(head);

    while (!st.empty())
    {
        head = st.top();
        st.pop();

        std::cout << head->val << ' ';

        if (head->right) st.push(head->right);
        if (head->left)  st.push(head->left);
    }

    std::cout << '\n';
}

// Left Root Right. 
void inorderTraverse(Node * head)
{
    if (!head) return;
    
    std::stack<Node *> st;

    // Stack is empty when resolving root's right subtree. 
    while (!st.empty() || head)
    {
        if (head)
        {
            st.emplace(head);
            head = head->left;
        }
        else
        {
            head = st.top();
            st.pop();
            std::cout << head->val << ' ';
            head = head->right;
        }
    }

    std::cout << '\n';
}

void postorderTraverse(Node * head)
{
    if (!head) return;

    std::stack<Node *> st;
    st.emplace(head);

    // head remains root node until a leaf node gets printed. 
    // After that, head denotes the previous node printed.     
    while (!st.empty())
    {
        Node * cur = st.top();

        if (cur->left && head != cur->left && head != cur->right)
        {
            // Has left subtree and left subtree unresolved.
            st.emplace(cur->left);
        }
        else if (cur->right && head != cur->right)
        {
            // Has right subtree and right subtree unresolved.
            st.emplace(cur->right);
        }
        else 
        {
            // Leaf node or all subtrees resolved.
            std::cout << cur->val << ' ';
            head = cur;
            st.pop();
        }
    }

    std::cout << '\n';
}

void preorderTraverseMorris(Node * root)
{
    while (root)
    {
        if (!root->left)
        {
            std::cout << root->val << ' ';
            root = root->right;
        }
        else
        {
            // prev->right == root iff. when this left subtree is traversed twice; 
            // this happens after root = root->right (!root->left). 
            Node * prev = root->left;
            while (prev->right && prev->right != root) prev = prev->right;

            if (!prev->right)
            {
                std::cout << root->val << ' ';
                prev->right = root;
                root = root->left;
            }
            else
            {
                prev->right = nullptr;
                root = root->right;
            }
        }
    }

    std::cout << '\n';
}

void inorderTraverseMorris(Node * root)
{
    while (root)
    {
        if (!root->left)
        {
            std::cout << root->val << ' ';
            root = root->right;
        }
        else
        {
            // prev->right == root iff. when this left subtree is traversed twice; 
            // this happens after root = root->right (!root->left). 
            Node * prev = root->left;
            while (prev->right && prev->right != root) prev = prev->right;

            if (!prev->right)
            {
                prev->right = root;
                root = root->left;
            }
            else
            {
                std::cout << root->val << ' ';
                prev->right = nullptr;
                root = root->right;
            }
        }
    }

    std::cout << '\n';
}

int main(int argc, char * argv[])
{
    //           0
    //      1         2
    //  3      4    5     6
    //    7  8    9  10      
    std::vector<Node> buf(11);
    buf[0] = {0, &buf[1], &buf[2]};
    buf[1] = {1, &buf[3], &buf[4]};
    buf[2] = {2, &buf[5], &buf[6]};
    buf[3] = {3, nullptr, &buf[7]};
    buf[4] = {4, &buf[8], nullptr};
    buf[5] = {5, &buf[9], &buf[10]};
    buf[6] = {6, nullptr, nullptr};
    buf[7] = {7, nullptr, nullptr};
    buf[8] = {8, nullptr, nullptr};
    buf[9] = {9, nullptr, nullptr};
    buf[10] = {10, nullptr, nullptr};
    Node * root = &buf[0];

    preorderTraverse(root);
    preorderTraverseMorris(root);

    inorderTraverse(root);
    inorderTraverseMorris(root);

    postorderTraverse(root);

    return EXIT_SUCCESS;
}
```



## 021 Merge Sort

```c++
// a[lo, hi), recursive version. 
void mergeSort(int * arr, int lo, int hi)
{
    if (hi < lo + 2) return;

    int mi = lo + ((hi - lo) >> 1);
    mergeSort(arr, lo, mi);
    mergeSort(arr, mi, hi);

    merge(arr, lo, mi, hi);
}

// a[lo, hi), iterative version. 
void mergeSort(int * a, int lo, int hi)
{
    if (hi < lo + 2) return;
    
    // Offset into a[0, hi). 
    a += lo;
    hi -= lo;
    tmp.resize(hi);
    
    for (int size = 1; size < hi; size <<= 1)
    {
        for (int i = 0, j, k; i < hi; i += (size << 1))
        {
            j = i + size;
            if (hi <= j) break;
            k = std::min(j + size, hi);
            merge(a, i, j, k);
        }
    }
}

// a[lo, hi)
void merge(int * a, int lo, int mi, int hi)
{
    std::copy(a + lo, a + mi, tmp.data() + lo);
    int * b = tmp.data() + lo;
    const int m = mi - lo;

    int * c = a + mi;
    const int n = hi - mi;

    for (int i = lo, j = 0, k = 0; j < m || k < n; )
    {
        if (j < m && (n <= k || b[j] <= c[k])) a[i++] = b[j++];
        if (k < n && (m <= j || c[k] <  b[j])) a[i++] = c[k++];
    }
}

// Helper space. 
std::vector<int> tmp;
```



## 022 Merge

1. 思考一个问题在大范围上的答案，是否等于，左部分的答案 + 右部分的答案 + 跨越左右产生的答案
2. 计算“跨越左右产生的答案”时，如果加上左、右各自有序这个设定，会不会获得计算的便利性
3. 如果以上两点都成立，那么该问题很可能被归并分治解决（话不说满，因为总有很毒的出题人）
4. 求解答案的过程中只需要加入归并排序的过程即可，因为要让左、右各自有序，来获得计算的便利性
- [LC 493. Reverse Pairs](https://leetcode.cn/problems/reverse-pairs/) 
  - Merge and regular merge sort. 
- [小和问题](https://www.nowcoder.com/practice/edfe05a1d45c4ea89101d936cac32469)
```c++
// 小和问题
// 假设数组 s = [ 1, 3, 5, 2, 4, 6 ]
// 在s[0]的左边所有 <= s[0]的数的总和为0
// 在s[1]的左边所有 <= s[1]的数的总和为1
// 在s[2]的左边所有 <= s[2]的数的总和为4
// 在s[3]的左边所有 <= s[3]的数的总和为1
// 在s[4]的左边所有 <= s[4]的数的总和为6
// 在s[5]的左边所有 <= s[5]的数的总和为15
// 所以s数组的“小和”为 : 0 + 1 + 4 + 1 + 6 + 15 = 27
// 给定一个数组arr，实现函数返回arr的“小和”
long long smallSum(int * arr, int lo, int hi) 
{
    if (hi < lo + 2) return 0;
    int mi = (lo + hi) / 2;
    return smallSum(arr, lo, mi) + smallSum(arr, mi, hi) + mergeSmallSum(arr, lo, mi, hi);
}

// 返回跨左右产生的小和累加和，左侧有序、右侧有序，让左右两侧整体有序
// arr[l...m] arr[m+1...r]
long long mergeSmallSum(int * arr, int lo, int mi, int hi) 
{
    // 统计部分
    long long ans = 0;
    
    for (int i = lo, j = mi, sum = 0; j < hi; ++j) 
    {
        while (i < mi && arr[i] <= arr[j]) 
        {
            sum += arr[i++];
        }

        ans += sum;
    }

    // 正常merge
    merge(arr, lo, mi, hi);

    return ans;
}
```



## 023 Quick Sort

- Dutch Flag style quick sort:
```c++
// a[lo, hi], recursive version. 
void quickSort(int * a, int lo, int hi)
{
    if (hi < lo + 2) return;

    auto [l, r] = partition(a, lo, hi - 1);
    quickSort(a, lo, l);
    quickSort(a, r + 1, hi);
}

// a[lo, hi], iterative version. 
void quickSort(int * a, int lo, int hi)
{
    if (hi < lo + 1) return;

    std::stack<std::pair<int, int>> st;
    st.emplace(lo, hi);

    while (!st.empty())
    {
        std::tie(lo, hi) = st.top();
        st.pop();
        auto [ll, rr] = partition(a, lo, hi);
        if (rr + 1 < hi) st.emplace(rr + 1, hi);
        if (lo < ll - 1) st.emplace(lo, ll - 1);
    }
}

// a[lo, hi]
std::pair<int, int> partition(int * a, int lo, int hi)
{
    int p = a[lo + std::rand() % (hi - lo + 1)];
    int mi = lo;

    while (mi <= hi)
    {
        if (a[mi] < p) std::swap(a[lo++], a[mi++]);
        else if (a[mi] == p) ++mi;
        else std::swap(a[hi--], a[mi]);
    }

    return {lo, hi};
}
```
- Legacy partition routine:
```c++
// a[lo, hi]
int partition(int * a, int lo, int hi)
{
    std::swap(a[lo], a[lo + std::rand() % (hi - lo + 1)]);
    int p = a[lo];

    while (lo < hi)
    {
        while (lo < hi && p < a[hi]) --hi;
        if (lo < hi) a[lo++] = a[hi];
        while (lo < hi && a[lo] < p) ++lo;
        if (lo < hi) a[hi--] = a[lo];
    }

    a[lo] = p;
    return lo;
}
```



## 024 Quick Select

- [LC 215 Kth Largest Element in an Array](https://leetcode.com/problems/kth-largest-element-in-an-array/)
```c++
// Pass in hi - lo - k for k-th largest element with non-decreasing partition. 
int quickSelect(int * a, int lo, int hi, int k)
{
    a += lo;
    hi -= lo;
    lo = 0;

    while (lo < hi)
    {
        auto [ll, rr] = partition(a, lo, hi - 1);
        
        if (k < ll)      hi = ll;
        else if (rr < k) lo = rr + 1;
        else             return a[k];
    }

    return -1;
}
```



## 025 Heap

```c++
// a[0..i) denotes a heap, push a[i] into heap. 
// 注意这里【不能】用 (i - 1) >> 1，移位的话 i == 0 时会溢出！
void pushHeap(int * a, int i)
{
    while (a[(i - 1) / 2] < a[i]) 
    {
        std::swap(a[(i - 1) / 2], a[i]);
        i = (i - 1) / 2;
    }
}

// i位置的数，变小了，又想维持大根堆结构
// 向下调整大根堆
// 当前堆的大小为size
void heapify(int * a, int i, int size)
{
    int l = 2 * i + 1;

    while (l < size)
    {
        // 有左孩子，l
		    // 右孩子，l + 1
		    // 评选，最强的孩子，是哪个下标的孩子
        int best = l + 1 < size && a[l] < a[l + 1] ? l + 1 : l;
        
        // 上面已经评选了最强的孩子，接下来，当前的数和最强的孩子之前，最强下标是谁
        best = a[i] < a[best] ? best : i;

        if (best == i) break;
        
        std::swap(a[best], a[i]);
        i = best;
        l = i * 2 + 1;
    }
}

// 从顶到底建立大根堆，O(n * logn)
// 依次弹出堆内最大值并排好序，O(n * logn)
// 整体时间复杂度O(n * logn)
void heapSort1(int * a, int n) 
{
    for (int i = 0; i < n; i++) pushHeap(a, i);
    
    for (int size = n; 1 < size; )
    {
        std::swap(a[0], --size);
        heapify(a, 0, size);
    }
}

// 从底到顶建立大根堆，O(n)
// 依次弹出堆内最大值并排好序，O(n * logn)
// 整体时间复杂度O(n * logn)
void heapSort2(int * a, int n) 
{
    for (int i = n - 1; 0 <= i; --i) heapify(a, i, n);

    for (int size = n; 1 < size; )
    {
        std::swap(a[0], --size);
        heapify(a, 0, size);
    }
}
```



## 027 堆结构常见题

- [LC 23. Merge k Sorted Lists](https://leetcode.com/problems/merge-k-sorted-lists/)
- [LC 2208. Minimum Operations to Halve Array Sum](https://leetcode.com/problems/minimum-operations-to-halve-array-sum/)
- [最多线段重合问题](https://www.nowcoder.com/practice/1ae8d0b6bb4e4bcdbf64ec491f63fc37)
```c++
std::sort(lines.cbegin(), lines.cend());
std::priority_queue<int, std::vector<int>, std::greater<int>> heap;
int ans = 0;
for (std::vector<int> & line : lines) 
{
    while (!heap.empty() && heap.top() <= line[0]) heap.pop();
    heap.emplace(line[1]);
    ans = std::max(ans, heap.size());
}
return ans;
```



## 028 基数排序 Radix Sort

- [LC 912. Sort an Array](https://leetcode.com/problems/sort-an-array/)
```c++
// a[lo, hi)
void radixSort(int * a, int lo, int hi) 
{
    // Offset indices into a[0, hi). 
    a += lo;
    hi -= lo;
    
    // Offset values into non-negative, radix sort, then offset back. 
    int minimum = *std::min_element(a, a + hi);
    for (int i = 0; i < hi; ++i) a[i] -= minimum;
    radixSortImpl(a, 0, hi);
    for (int i = 0; i < hi; ++i) a[i] += minimum;
}

// a[lo, hi), MUST be all non-negative. 
void radixSortImpl(int * a, int lo, int hi, int base = 10)
{
    cnt.resize(base);
    
    // Offset into a[0, hi). 
    a += lo;
    hi -= lo;
    tmp.resize(hi);

    // Number of bits in radix base. 
    int bits = 0;
    for (int x = *std::max_element(a, a + hi); 0 < x; x /= base) ++bits;
    
    for (int offset = 1; 0 < bits; offset *= base, --bits)
    {
        // Count bits into culmulative sum. 
        // Block write-back in REVERSE order for stability. 
        std::fill_n(cnt.data(), base, 0);
        for (int i = 0; i < hi; ++i) ++cnt[(a[i] / offset) % base];
        for (int i = 1; i < base; ++i) cnt[i] += cnt[i - 1];
        for (int i = hi - 1; 0 <= i; --i) tmp[--cnt[(a[i] / offset) % base]] = a[i];
        std::copy_n(tmp.data(), hi, a);
    }
}

// Helper space. 
std::vector<int> cnt;
std::vector<int> tmp;
```

## 003/030/031 二进制和位运算 异或运算的骚操作 位运算的骚操作 Bitwise XOR

### Bitwise AND

- `x & (x - 1)`: Unset lowest 1 bit;
  - Usage: Bin count, etc. 
- Brian Kernighan Algorithm: `x & (-x)` or `x & (~x + 1)`
  - Extract lowest 1 bit (unset all other bits);
  - Usage e.g.: Binary Indexed Trees. 
- If `n` is power of 2, then `x % n` equals `x & (n - 1)`. 

### Bitwise XOR

- Basics
  - Carry-less addition; 
  - Communicative; 
  - Associative. 
  - XOR zero equals self: `x ^ 0 == x`;
  - XOR self equals zero: `x ^ x == 0`;
- `swap`: `{ a ^= b; b ^= a; a ^= b; }` 
  - Note that this expression is **wrong** when `&a == &b`!
- `flip` or `toggle` a bit: `x ^ 1`
  - `x` must be either `0` or `1`!
  - `0 -> 1`, `1 -> 0`
- 不用任何判断语句和比较操作，返回两个数的最大值
```c++
int getMax(int a, int b)
{
    // c可能是溢出的
    int c = a - b;
    // a的符号
    int sa = sign(a);
    // b的符号
    int sb = sign(b);
    // c的符号
    int sc = sign(c);
    // 判断A和B，符号是不是不一样，如果不一样diffAB=1，如果一样diffAB=0
    int diffAB = sa ^ sb;
    // 判断A和B，符号是不是一样，如果一样sameAB=1，如果不一样sameAB=0
    int sameAB = diffAB ^ 1;
    int returnA = diffAB * sa + sameAB * sc;
    int returnB = returnA ^ 1;
    return a * returnA + b * returnB;
}
```

### General Bitwise Operations

- [LC 231. Power of Two](https://leetcode.com/problems/power-of-two/)
- [LC 326. Power of Three](https://leetcode.com/problems/power-of-three/)
- 返回大于等于n的最小的2某次方，把n最高位1开始往后所有bit全部刷成1
```c++
// 已知n是非负数
// 返回大于等于n的最小的2某次方
// 如果int范围内不存在这样的数，返回整数最小值
int near2power(int n) 
{
    if (n <= 0) return 1;
    --n;
    
    // 把n最高位1开始往后所有bit全部刷成1
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    
    return n + 1;
}
```
- [LC 201. Bitwise AND of Numbers Range](https://leetcode.com/problems/bitwise-and-of-numbers-range/)
- [LC 190. Reverse Bits](https://leetcode.com/problems/reverse-bits/)
- [LC 461. Hamming Distance](https://leetcode.com/problems/hamming-distance/)



## 032 位图 Bit Set

- [LC 2166. Design Bitset](https://leetcode.com/problems/design-bitset/)



## 033 位运算实现加减乘除

- [位运算实现加减乘除](https://github.com/algorithmzuo/algorithm-journey/blob/main/src/class033/BitOperationAddMinusMultiplyDivide.java)



## 034 链表高频题目和必备技巧

- Slow/fast Pointers:
  - Floyd's algorithm to find 1st node in circle;
  - Find mid node of a forward list; 
  - ...
- [LC 160. Intersection of Two Linked Lists](https://leetcode.com/problems/intersection-of-two-linked-lists/)
  - Two pointers EQUAL stepsize. 
  - When one finishes, go to head of THE OTHER. 
  - Meets at intersection node because of equal steps taken. 
- [LC 25. Reverse Nodes in k-Group](https://leetcode.com/problems/reverse-nodes-in-k-group/)
- [LC 138. Copy List with Random Pointer](https://leetcode.com/problems/copy-list-with-random-pointer/)
  - Interweave new nodes into old list and unweave them. 
- [LC 234. Palindrome Linked List](https://leetcode.com/problems/palindrome-linked-list/)
  - Reverse the latter half of the list and loop, finally restore. 
- [LC 142. Linked List Cycle II](https://leetcode.com/problems/linked-list-cycle-ii/)
- [LC 148. Sort List](https://leetcode.com/problems/sort-list/)
  - Bottom-up merge sort. 



## 035 数据结构设计高频题

- [LC 146. LRU Cache](https://leetcode.com/problems/lru-cache/)
  - Nodes stored as linked list, LRU nodes moved to front;
  - Hashmap value to node for indexing. 
- [LC 380. Insert Delete GetRandom O(1)](https://leetcode.com/problems/insert-delete-getrandom-o1/)
  - Store as a vector, with an unordered_map index for duplicate lookup. 
  - Delete: Swap with back element and remove
- [LC 381. Insert Delete GetRandom O(1) - Duplicates allowed](https://leetcode.com/problems/insert-delete-getrandom-o1-duplicates-allowed/)
  - Store as a vector, with an `unordered_map<int, unordered_map<int>>` for duplicate lookup. 
  - Delete: Swap any occurance with back element and remove. 
  - Special care MUST be taken for identical val/bak values!
    - Doing push/pops on the same index HashSet will be buggy!
- [LC 295. Find Median from Data Stream](https://leetcode.com/problems/find-median-from-data-stream/)
  - One small heap and one large heap. 
- [LC 895. Maximum Frequency Stack](https://leetcode.com/problems/maximum-frequency-stack/)
  - One data storage `unordered_map<int, stack<int>>` of Frequency, f-th occurances of elements. 
  - Another index `unordered_map<int, int>` stores frequencies of values. 
- [LC 432. All O`one Data Structure](https://leetcode.com/problems/all-oone-data-structure/)
  - Increase/decrease frequencies of elements by 1;
  - Get one of the elements with max/min frequency. 
  - Data storage: `list<int, string>` of frequency, value, in ascending order. 
  - Index: `unordered_map<string, list<int, string>::iterator>`, index to list node. 
  - Inc/dec by 1: Move to neighbors, possibly merging or freeing. 
  - **Don't** show-off. 
    - Just go if/else blocks even if this duplicates code; 
    - First make the mind clear and the program run. 
- [LC 716. Max Stack](https://leetcode.com/problems/max-stack/)
  - Implementation ONE
    - `Heap<val, timeStamp>` and `Stack<val, timeStamp>`, with lazy removal tag `HashSet<timeStamp>`. 
    - For each push/pop/peekMax/popMax, first remove all top lazy-tagged items. 
  - Implementation TWO
    - `List<val, timeStamp>` and `TreeSet<Pair<val, timeStamp>, ListIterator>`
    - Note that reversed iterators are shifted!!!
      - `*tree.rbegin() == *std::prev(tree.end())`, but
      - `tree.rbegin().base() != std::prev(tree.end())`!!!
      - `tree.rbegin().base() == tree.end()`!!! 
- [LC 460. LFU Cache](https://leetcode.com/problems/lfu-cache/)
  - Data: `HashSet<Freq, List<Key, Value>>`;
  - Index: `HashSet<Key, Pair<Freq, ListIter>>`;
  - Deal with `insert(k, v, f)` and `erase(k)` and reuse.
- [LC 2336. Smallest Number in Infinite Set](https://leetcode.com/problems/smallest-number-in-infinite-set/)
  - (1) HashSet and Heap (both added numbers) 
  - (2) TreeSet (added numbers)



## 036 二叉树高频题目-上-不含树型DP

- [LC 102. Binary Tree Level Order Traversal](https://leetcode.com/problems/binary-tree-level-order-traversal/)
  - Binary tree level size == Size of BFS queue (Solve all nodes in this level in a for loop).  
- [LC 103. Binary Tree Zigzag Level Order Traversal](https://leetcode.com/problems/binary-tree-zigzag-level-order-traversal/)
- [LC 662. Maximum Width of Binary Tree](https://leetcode.com/problems/maximum-width-of-binary-tree/)
- [LC 104. Maximum Depth of Binary Tree](https://leetcode.com/problems/maximum-depth-of-binary-tree/)
- [LC 297. Serialize and Deserialize Binary Tree](https://leetcode.com/problems/serialize-and-deserialize-binary-tree/)
  - 二叉树可以通过先序、后序或者按层遍历的方式序列化和反序列化
  - 但是，二叉树**无法**通过中序遍历的方式实现序列化和反序列化
  - 因为不同的两棵树，可能得到同样的中序序列，即便补了空位置也可能一样。
  - 比如`1 <- 2 -> NULL`和`NULL <- 1 -> 2`
  - 补足空位置的中序遍历结果都是`null, 1, null, 2, null`
- [LC 105. Construct Binary Tree from Preorder and Inorder Traversal](https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/)
  - A binary tree is incomplete iff. 
    - We see a node with right chlid but without left child, or:
    - We are at bottom level (have seen a node with 1 child or none), and see a node with child(ren). 
- [LC 222. Count Complete Tree Nodes](https://leetcode.com/problems/count-complete-tree-nodes/)
  - Binary search on bottom level
  - Whether the k-th node on bottom level exists: Iterate k's bin expr from msb to lsb, 0 left 1 right. 



## 037二叉树高频题目-下-不含树型DP

- [LC 236. Lowest Common Ancestor of a Binary Tree](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/)
  - Recursion
  - Iteration, `std::stack<TreeNode *, ChildrenVisited>`, pop iff.  all 2 subtrees visited. 
```c++
TreeNode * lca(TreeNode * root, TreeNode * p, TreeNode * q)
{
    if (!root || root == p || root == q) return root;
    TreeNode * ll = lca(root->left, p, q);
    TreeNode * rr = lca(root->right, p, q);
    if (ll && rr) return root;
    if (!ll && !rr) return nullptr;
    return ll ? ll : rr;
}
```
- [LC 235. Lowest Common Ancestor of a Binary Search Tree](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree/)
```c++
while (root)
{
    if (p->val < root->val && q->val < root->val) root = root->left;
    else if (root->val < p->val && root->val < q->val) root = root->right;
    else return root;
}
```
- [LC 113. Path Sum II](https://leetcode.com/problems/path-sum-ii/)
- [LC 110. Balanced Binary Tree](https://leetcode.com/problems/balanced-binary-tree/)
- [LC 98. Validate Binary Search Tree](https://leetcode.com/problems/validate-binary-search-tree/)
  - Compare with threholds of type `long long` in case single node `INT_MAX`. 
- [LC 669. Trim a Binary Search Tree](https://leetcode.com/problems/trim-a-binary-search-tree/)
- [LC 337. House Robber III](https://leetcode.com/problems/house-robber-iii/)



## 038 常见经典递归过程解析

- [LC 90. Subsets II](https://leetcode.com/problems/subsets-ii/)
  - Dups from not taking same number of eles in dup chunk. 
  - E.g., `1 2 2 3 ...`, the following decisions results in a same subset:
    - Not taking 1st 2 and taking 2nd 2; 
    - Taking 1st 2 and not taking 2nd 2. 
  - For each ele:
    - Take ele, and backtrack;
    - Ignore 1, 2, ..., len(chunk) ele(s) in this dup chunk, and backtrack. 
- [LC 46. Permutations](https://leetcode.com/problems/permutations/)
- [LC 47. Permutations II](https://leetcode.com/problems/permutations-ii/)
  - Dups from backtracking with identical curr element. 
- Sort stack with recursion and O(1) aux space:
```c++
#include <bits/stdc++.h>

#include <fmt/core.h>
#include <fmt/ranges.h>

template <typename T, typename BinaryOperation>
T accumulate(std::stack<T> & stk, T init, BinaryOperation op)
{
    if (stk.empty()) return init;
    T top = stk.top();
    stk.pop();
    init = accumulate(stk, op(init, top), op);
    stk.emplace(top);
    return init;
}

template <typename T, typename BinaryOperation>
T accumulateDepth(std::stack<T> & stk, int depth, T init, BinaryOperation op)
{
    if (depth <= 0 || stk.empty()) return init;
    T top = stk.top();
    stk.pop();
    init = accumulateDepth(stk, depth - 1, op(init, top), op);
    stk.emplace(top);
    return init;
}

// 从栈当前的顶部开始，往下数deep层，已知最大值是max，出现了k次
// 请把这k个最大值沉底，剩下的数据状况不变
void down(std::stack<int> & stk, int depth, int maxi, int k)
{
    if (depth <= 0)
    {
        for (int i = 0; i < k; i++) stk.emplace(maxi);
    }
    else
    {
        int top = stk.top();
        stk.pop();
        down(stk, depth - 1, maxi, k);
        if (top != maxi) stk.push(top);
    }
}

// 用递归函数排序栈
// 栈只提供push、pop、empty三个方法
// 请完成无序栈的排序，要求排完序之后，从栈顶到栈底从小到大
// 只能使用栈提供的push、pop、empty三个方法、以及递归函数
// 除此之外不能使用任何的容器，数组也不行
// 就是排序过程中只能用：
// 1) 栈提供的push、pop、empty三个方法
// 2) 递归函数，并且返回值最多为单个整数
void sort(std::stack<int> & stk)
{
    // 返回栈的深度
	// 不改变栈的数据状况
    int depth = accumulate(stk, 0, [](int d, int) { return 1 + d; });

    while (0 < depth)
    {
        // 从栈当前的顶部开始，往下数deep层
	    // 返回这deep层里的最大值
        int maxi = accumulateDepth(stk, depth, std::numeric_limits<int>::min(), [](int a, int b) 
        {
            return std::max(a, b); 
        });

        // 从栈当前的顶部开始，往下数deep层，已知最大值是max了
	    // 返回，max出现了几次，不改变栈的数据状况
        int k = accumulateDepth(stk, depth, 0, [maxi](int a, int b) 
        {
            return a + (b == maxi); 
        });

        down(stk, depth, maxi, k);
        depth -= k;
    }
}

int main(int argc, char * argv[])
{
    std::deque<int> deq {1, 2, 3, 3, 3, 4, 5, 6, 10, 10};
    std::stack<int> stk(deq);
    sort(stk);

    fmt::print("{}\n", stk);
    // Prints: [10, 10, 6, 5, 4, 3, 3, 3, 2, 1]

    return EXIT_SUCCESS;
}
```



## 039 嵌套类问题的递归解题套路

- Overall Process:
  - Declare global index `i`;
  - Recursive function `f(s)`: 
    - Comsume `s[i...]` until termination condition (end-of-input or closing brakets): 
      - `for ( ; i < s.size() && s[i] != ')'; ++i) { ... }`;
    - Return result for this chunk;
    - Updates global index `i` for successors. 
- [LC 394. Decode String](https://leetcode.com/problems/decode-string/)
- [LC 726. Number of Atoms](https://leetcode.com/problems/number-of-atoms/)
- [LC 772. Basic Calculator III](https://leetcode.com/problems/basic-calculator-iii/)
```c++
class Solution
{
public:
    int calculate(std::string s)
    {
        i = 0;
        return f(s);
    }

private:
    int f(const std::string & s)
    {
        auto n = static_cast<const int>(s.size());

        std::vector<char> oper;
        std::vector<int> oprd;

        int cur = 0;

        for ( ; i < n && s[i] != ')'; ++i)
        {
            int c = s[i];

            if (std::isdigit(c))
            {
                cur = cur * 10 + c - '0';
            }
            else if (c != '(')  // + - * /
            {
                // Note: ')' will be consumed automatically
                // by the for update after recursive f() call. 
                push(oper, oprd, c, cur);
                cur = 0;
            }
            else  // (
            {
                ++i;
                cur = f(s);
            }
        }

        // Consume the last operand. 
        // The operator could be anything. 
        push(oper, oprd, '+', cur);

        return compute(oper, oprd);
    }

    void push(std::vector<char> & oper, std::vector<int> & oprd, char op, int cur)
    {
        if (oprd.empty() || oper.back() == '+' || oper.back() == '-')
        {
            oper.emplace_back(op);
            oprd.emplace_back(cur);
        }
        else
        {
            oprd.back() = oper.back() == '*' ? oprd.back() * cur : oprd.back() / cur;
            oper.back() = op;
        }
    }

    int compute(std::vector<char> & oper, std::vector<int> & oprd)
    {
        auto n = static_cast<const int>(oprd.size());
        int ans = oprd[0];

        for (int i = 1; i < n; ++i)
        {
            ans = oper[i - 1] == '+' ? ans + oprd[i] : ans - oprd[i];
        }

        return ans;
    }

    // Current index-of-processing of string s. 
    int i = 0;
};
```



## 040 N皇后问题-重点是位运算的版本

- [LC 52. N-Queens II](https://leetcode.com/problems/n-queens-ii/)



## 041 最大公约数、同余原理

- Modulo: Result modulo `p`. Replace every operation with: 
  - Plus: `a + b -> (a + b) % p`
  - Minus: `a - b -> (a - b + p) % p`
  - Multiplies: `a * b -> (a * b) % p`
  - Divides: TODO
- Eulicid's Algorithm
```c++
// int gcd(int a, int b)
// {
//     if (a < b) std::swap(a, b);

//     for (int t; b; )
//     {
//         t = b;
//         b = a % b;
//         a = t;
//     }

//     return a;
// }

int gcd(int a, int b)
{
    return b == 0 ? a : gcd(b, a % b);
}

int lcm(int a, int b)
{
    return a / gcd(a, b) * b;
    // OR 
    // return static_cast<long long>(a * b) / gcd(a, b);
}
```
- [LC 878. Nth Magical Number](https://leetcode.com/problems/nth-magical-number/)


## 042 对数器打表找规律的技巧

- TODO



## 043 根据数据量猜解法的技巧-天字第一号重要技巧

- Time Constraint:
  - 1s for C/C++;
  - 1-2s for Java/Python/Go...
- Corresponding number of constant instructions:
  - 1e7-1e8. 
  - Regardless of platform, CPU, ...
- Required complexity with respect to input size:
  - `> 1e8`: **O(log n)**
  - `<= 1e7`: **O(n)**, O(log n)
  - `<= 1e6`: **O(n log n)**, O(n), O(log n)
  - `<= 1e5`: **O(n sqrt n)**, O(n log n), O(n), O(log n)
  - `<= 5000`: **O(n^2)**, O(n sqrt n), O(n log n), O(n), O(log n)
  - `<= 25`: **O(2^n)**, O(n^2), O(n sqrt n), O(n log n), O(n), O(log n)
  - `<= 11`: **O(n!)**, O(2^n), O(n^2), O(n sqrt n), O(n log n), O(n), O(log n)
- [LC 9. Palindrome Number](https://leetcode.com/problems/palindrome-number/)
- [LC 906. Super Palindromes](https://leetcode.com/problems/super-palindromes/)


## 044 前缀树原理和代码详解 Trie

- [Trie](../../notes/dsa/Trie.cpp)
- [LC 421. Maximum XOR of Two Numbers in an Array](https://leetcode.com/problems/maximum-xor-of-two-numbers-in-an-array/)
- [LC 212. Word Search II](https://leetcode.com/problems/word-search-ii/)
  - Trie all words to search;
  - Prune DFS with this Trie. 



## 046 构建前缀信息的技巧-解决子数组相关问题 Prefix Sum / Cumulative Sum

- 1D Prefix Sum Implementation Considerations: 
  - 0-indexed;
  - **Zero padding on the left side**;
  - `ps` array indices offset the original by +1. 
```c++
std::vector<int> arr = ...;
auto n = static_cast<const int>(arr.size());
std::vector<int> ps(n + 1, 0);

for (int i = 0; i < n; ++i) ps[i + 1] = ps[i] + arr[i];   // Option 1.

std::inclusive_scan(
        arr.cbegin(), arr.cend(), 
        ps.begin() + 1);                                  // Option 2.

std::inclusive_scan(
        arr.cbegin(), arr.cend(), 
        ps.begin() + 1, 
        std::plus<>(), 0LL);                              // Option LL1. 

std::transform_inclusive_scan(
        arr.cbegin(), arr.cend(), 
        ps.begin() + 1, 
        std::plus<>(), 
        [](int x) { return static_cast<long long>(x); },
        0LL);                                             // Option LL2.
```
- [STL Considerations for Prefix Sum](./implementation_notes_stl.md)
  - `std::inclusive_scan` seems to be the best (requires `c++17`);
  - Leave a zero padding on index 0. `ps[i + 1] = ps[i] + arr[i];`
- [LC 303. Range Sum Query - Immutable](https://leetcode.com/problems/range-sum-query-immutable/)
- [LC 560. Subarray Sum Equals K](https://leetcode.com/problems/subarray-sum-equals-k/)
  - Note that the at most k sliding window won't work for this problem as element may be negative. 
  - [LC 930. Binary Subarrays With Sum](https://leetcode.com/problems/binary-subarrays-with-sum/)
  - Prefix sum could be computed with rolling variable manner in O(1) space. 
- [未排序数组中所有子数组中正数与负数个数相等的最长子数组的长](https://www.nowcoder.com/practice/545544c060804eceaed0bb84fcd992fb)
  - Treat positive as 1 and negative as -1. 
  - A.k.a. Subarray Sum Equals 0. 
- [LC 1124. Longest Well-Performing Interval](https://leetcode.com/problems/longest-well-performing-interval/)
  - Array of -1 and 1, ask for the longest subarray with a positive sum. 
  - `HashMap<PrefixSum, LowestIndex>`
  - When sum go non-positive, the sum must have decreased by 1. 
- [LC 1590. Make Sum Divisible by P](https://leetcode.com/problems/make-sum-divisible-by-p/)
  - Prefix Sum modulo p, HashMap. 
- [LC 1371. Find the Longest Substring Containing Vowels in Even Counts](https://leetcode.com/problems/find-the-longest-substring-containing-vowels-in-even-counts/)
  - Prefix sum (no-carry, xor) of state compression binary mask, HashMap (or bucket). 



## 047 一维差分与等差数列差分

- 一维差分
  - 和一维前缀和互为逆运算
    - 前缀和：区间查询，但不支持更新。
  - 解决的问题
    - 区间更新，但要求所有更新都完成后才能离线查询（须O(n)预处理时间）；
    - 单点更新但在线查询：树状数组 Binary Indexed Tree (BIT)；
    - 区间更新但在线查询：线段树 Segment Tree。
  - Implementation considerations:
    - **0-indexed**;
    - **Zero padding on the ight side**:
```c++
std::vector<int> arr = ...;
auto n = static_cast<const int>(arr.size());
std::vector<int> diff(n + 1, 0);

// 0-indexed interval [a, b]. 
auto add = [&diff](int a, int b, int k) mutable
{
    diff[a] += k;
    diff[b + 1] -= k;
};

// Prefix-sum diff back into the original. 
for (int i = 1; i <= n; ++i) diff[i] += diff[i - 1];
```
- [LC 1109. Corporate Flight Bookings](https://leetcode.com/problems/corporate-flight-bookings/)
- 等差数列差分
  - 区间更新，`[l, r, s, e]`，对`arr[l...r]`加上一个首项`s`末项`e`的等差数列
  - 设公差为`d`，则原数组等于一阶差分的前缀和，而一阶差分等于二阶差分的前缀和。
  - 二阶差分操作：
```c++
// 2nd-order difference array. 
std::vector<long long> diff(n + 1, 0);

auto set = [&diff](int l, int r, int s, int e, int d) mutable
{
    diff[l] += s;
    diff[l + 1] += d - s;
    diff[r + 1] -= d + e;
    diff[r + 2] += e;
}

// Turn into 1st-order difference, then to vanilla.
for (int i = 1; i <= n; ++i) diff[i] += diff[i - 1];
for (int i = 1; i <= n; ++i) diff[i] += diff[i - 1];
```
```
E.g., l = 1, r = 7, s = 4, e = 16, then:

0  1   2  3   4   5   6   7    8   9    INDEX

0  4   6  8  10  12  14  16    0   0    MODIFICATION

0  4  -2  0   0   0   0   0  -18  16    DIFF2

0  4   2  2   2   2   2   2  -16   0    DIFF1

0  4   6  8  10  12  14  16    0   0    RESTORED MODIFICATION

  l l+2                    r r+1    r+2 r+3
0 s s+d s+2d s+3d ... s+kd e 0      0   0
0 s d   d    d    ... d    d -e     0   0
0 s d-s 0    0    ... 0    0 -(e+d) e   0
```
- [洛谷 P4231 三步必杀](https://www.luogu.com.cn/problem/P4231)



## 048 二维前缀和、二维差分、离散化技巧

- 2D Prefix Sum:
  - **0-indexed**;
  - **Zero padding on the left and the top boundaries**;
  - **`ps`'s (i, j) indices offset the original by (+1, +1)**. 
```c++
const std::vector<std::vector<int>> & arr = ...;
auto m = static_cast<const int>(arr.size());
auto n = static_cast<const int>(arr.front().size());
std::vector ps(m + 1, std::vector<int>(n + 1, 0));

for (int i = 0; i < m; ++i)
{
    for (int j = 0; j < n; ++j)
    {
        ps[i + 1][j + 1] = arr[i][j] + ps[i + 1][j] + ps[i][j + 1] - ps[i][j];
    }
}
```
- 2D Difference: 
  - **1-indexed**;
    - Note this differs from 1D difference (which is 0-indexed); 
  - **Zero padding on all four boundaries**:
    - Note this differs from 1D difference; 
  - Necessity:
    - For 1D, the left element needs no operation at all;
    - For 2D, the top and left boundary elements *needs* prefix sum operations too;
    - Thus left and top paddings, and 1-indexing, are needed. 
```c++
const std::vector<std::vector<int>> & arr = ...;
auto m = static_cast<const int>(arr.size());
auto n = static_cast<const int>(arr.front().size());
std::vector diff(m + 2, std::vector<int>(n + 2, 0));

// Add 1-indexed rectangle top-left (a, b) -> bottom-right (c, d) by k. 
auto add = [&diff](int a, int b, int c, int d, int k = 1) mutable
{
    diff[a][b] += k;
    diff[c + 1][b] -= k;
    diff[a][d + 1] -= k;
    diff[c + 1][d + 1] += k;
};

// Prefix-sum diff array back into the original. 
for (int i = 1; i <= m; ++i)
{
    for (int j = 1; j <= n; ++j)
    {
        diff[i][j] += diff[i - 1][j] + diff[i][j - 1] - diff[i - 1][j - 1];
    }
}
```
- [LC 304. Range Sum Query 2D - Immutable](https://leetcode.com/problems/range-sum-query-2d-immutable/)
- [LC 1139. Largest 1-Bordered Square](https://leetcode.com/problems/largest-1-bordered-square/)
- [洛谷 P3397 地毯](https://www.luogu.com.cn/problem/P3397)
- [LC 2132. Stamping the Grid](https://leetcode.com/problems/stamping-the-grid/)
  - 2D Prefix sum for quick testing whether a location is stampable; 
  - Conv all stampable locations, 2D Difference.  
- [LCP 74. 最强祝福力场](https://leetcode.cn/problems/xepqZ5/)
  - Discretize (sort x, y coordinates and take their ranks);
  - 2D Difference. 



## 049 滑动窗口技巧与相关题目

- 滑动窗口
  - 维持左、右边界都不回退的一段范围，来求解很多**子数组（串）的相关问题**
  - 求解大流程：**求子数组在** **每个位置** 开头 或 **结尾** **情况下的答案**（开头还是结尾在于个人习惯）
    - 问 **最长** 窗口：外围右移 `ll`，**内部固定 `ll`** 右移 `rr`，反过来不行
    - 问 **最短** 窗口：外围右移 `rr`，**内部固定 `rr`** 右移 `ll`
  - 滑动过程：滑动窗口可以用 简单变量 或者 结构 来 维护信息
    - 最外层递增`rr`, 内层固定`rr`，滑动或者计算`ll`:
      - `if (counter[arr[rr]]++ == 0) ++info;`
      - `while (ll <= rr && info is valid) { if (--counter[arr[ll++]] == 0) invalidate info; }`
  - 滑动窗口的关键：找到 范围 和 答案指标 之间的 单调性关系（类似贪心）
  - 滑动窗口维持 **最大值** 或者 **最小值** 的更新结构：**单调队列**
- [209. Minimum Size Subarray Sum](https://leetcode.com/problems/minimum-size-subarray-sum/)
- [3. Longest Substring Without Repeating Characters](https://leetcode.com/problems/longest-substring-without-repeating-characters/)
- [1234. Replace the Substring for Balanced String](https://leetcode.com/problems/replace-the-substring-for-balanced-string/)
  - Turn into LC 76. Min Window Substr. 
- [1759. Count Number of Homogenous Substrings](https://leetcode.com/problems/count-number-of-homogenous-substrings/)
  - Math Linspace Sum: `aaa...a` of length `k` has `sum([1...k])` homogeneous substrings. 
- [2958. Length of Longest Subarray With at Most K Frequency](https://leetcode.com/problems/length-of-longest-subarray-with-at-most-k-frequency/)
- [560. Subarray Sum Equals K](https://leetcode.com/problems/subarray-sum-equals-k/)
  - Note that this could be done only with prefix sum + HashMap method. 
  - Prefix sum could be computed with rolling number manner in O(1) space. 
  - Note that (b) for 930 does **not** work for this problem!!!
- [2444. Count Subarrays With Fixed Bounds](https://leetcode.com/problems/count-subarrays-with-fixed-bounds/)
- **"At most k"/"At least k"**
  - Number of subarrays with at most/least k something, count num ending at `rr` for valid window. 
    - "At most": 
      - A valid subarray could start from `[ll...rr]` and end at `rr`. 
      - `ans += rr - ll + 1`. 
    - "At least" (Essentially equivalant to "at most"): 
      - A valid subarray could start from `[0...ll)` and end at `rr`. 
      - `ans += ll` (`all (rr - 0 + 1) - at most (rr - ll + 1)`). 
  - [930. Binary Subarrays With Sum](https://leetcode.com/problems/binary-subarrays-with-sum/)
  - [713. Subarray Product Less Than K](https://leetcode.com/problems/subarray-product-less-than-k/)
  - [992. Subarrays with K Different Integers](https://leetcode.com/problems/subarrays-with-k-different-integers/description/)
    - Counter constituted by non-negative elements, could set minus. 
    - Sliding Window At Most k Distincts MINUS At Most k - 1 Distincts. 
  - [2962. Count Subarrays Where Max Element Appears at Least K Times](https://leetcode.com/problems/count-subarrays-where-max-element-appears-at-least-k-times/)
    - Sliding Window AND Monotonic Queue. 
- **"Needs k somewhat"**
  - [76. Minimum Window Substring](https://leetcode.com/problems/minimum-window-substring/)
  - Bucket count all target chars `needs`;
  - Maintain `needed = t.size()`:
    - `0 < needs[rr]--`, then `rr` is needed, `--needed`;
    - While `ll <= rr && needed == 0` move `ll`, `needs[s[ll++]]++ == 0` then `++needed`.



## 050 双指针技巧与相关题目

- 双指针
  - 同数组内：
    - 快慢指针：Floyd's Cycle Detection Algorithm，One-pass get last-k node in forward list
    - 相向指针：2-Sum，Trapping Rain Water，Container with Most Water，First Missing Positive
    - 同向指针：这个叫滑动窗口……
  - 两个数组内：
    - 同向指针：[LC 475. Heaters](https://leetcode.com/problems/heaters/)
- [LC 287. Find the Duplicate Number](https://leetcode.com/problems/find-the-duplicate-number/)
  - Floyd's algorithm (2 pointers cycle detection)
  - Array as hash (aka cyclic sort)
    - Let `nums[i]` store `i`;
    - `while (nums[0] != nums[nums[0]]) std::swap(nums[0], nums[nums[0]]);`
  - Negative marking (given that all elements are positive)
  - Bit count
    - Count total number of 1-bits for range `[1...n]`, 
    - compare with that for nums, 
    - the duplicate is bits where `bc1n < bcNums`.
- [LC 42. Trapping Rain Water](https://leetcode.cn/problems/trapping-rain-water/)
- [LC 881. Boats to Save People](https://leetcode.com/problems/boats-to-save-people/)
- [LC 11. Container With Most Water](https://leetcode.com/problems/container-with-most-water/)
- [LC 475. Heaters](https://leetcode.com/problems/heaters/)
  - Two pointers, `houses` and `heaters` each has one pointer. 
  - `heaters` pointer `j` points to 1st heater on the right of `houses[i]` (inclusive). 
- [LC 41. First Missing Positive](https://leetcode.com/problems/first-missing-positive/)
  - Implementation 1
    - Input array sign bit as hash. 
    - Special care for `[0...n-1]` (lacks `n`) and `[1...n]` (lacks `n + 1`).
  - Implementation 2
    - Two pointers. Assume `[ll, rr]` contains all numbers in `[ll + 1, rr + 1]`. 
    - `[ll, rr]` current interval, `[0, ll - 1] == i + 1`, `(rr...n - 1]` storestrash.  



## 051 二分答案法与相关题目

- 二分的是【答案】
  - 求的答案是一个数；
  - 这个数能估计出上界和下界；
  - 给定一个候选答案，能 `O(n)` 判断出它是大了还是小了。
- [875. Koko Eating Bananas](https://leetcode.com/problems/koko-eating-bananas/)
  - Bin search on `k` (how many bananas to eat per hour) `O(log m)`, `max(piles) == m <= 1e9`. 
  - For each `k` decide to go left/right in `O(n)` time, `piles.size() == n <= 1e4`. 
  - `O(n log m)` in total, valid. 
- [410. Split Array Largest Sum](https://leetcode.com/problems/split-array-largest-sum/)
  - Bin Search on the Result Largest Sum.
  - Given this largest sum, calculate how many segments we need to split original array into. 
- [719. Find K-th Smallest Pair Distance](https://leetcode.com/problems/find-k-th-smallest-pair-distance/)
  - Bin Search AND Sliding Window At Most K
- [1760. Minimum Limit of Balls in a Bag](https://leetcode.com/problems/minimum-limit-of-balls-in-a-bag/)
- [2064. Minimized Maximum of Products Distributed to Any Store](https://leetcode.com/problems/minimized-maximum-of-products-distributed-to-any-store/)
- [2141. Maximum Running Time of N Computers](https://leetcode.com/problems/maximum-running-time-of-n-computers/)
- [2226. Maximum Candies Allocated to K Children](https://leetcode.com/problems/maximum-candies-allocated-to-k-children/)
- 服务员问题
  - 描述：
    - 有 `n` 个服务员，第 `i` 个服务员服务完一个客人所需时间为 `waiter[i]`；
    - 现有 `m` 个客人在等位，你排在这 `m` 人后面，你最少要等多久？
  - 限制：
    - `1 <= n <= 1e3`
    - `1 <= m <= 1e9`
  - 解法：
    - 二分等待时间，下界 `0`，上界 `m * min(waiter)`；
    - 对每一个等位时间，能 `O(n)` 算出这段时间里最多能服务多少客人。
- 打怪兽问题
  - 描述：
    - 有一个怪兽，有 `hp` 单位的血量；
    - 每一回合 `i`，你可以平A一刀，造成 `f[i]` 单位伤害；或者下毒，这一回合不造成伤害，从下一回合开始，每回合造成 `g[i]` 单位伤害；
    - 不同回合多次下毒造成的伤害可以叠加；
    - 你最多可以行动 `n` 个回合，如果 `n` 回合结束后怪兽还没死，你也不能继续行动；但如果怪兽此前中毒了，毒属性伤害仍继续生效；
    - 问最快几回合能把怪兽打死？
  - 限制：
    - `1 <= hp <= 1e9`
    - `1 <= n <= 1e5`
    - `1 <= f[i], g[i] <= 1e9`
  - 解法：
    - 二分所需回合数，下界 `1`，上界 `hp + 1`；
    - 对每一个回合数，可以 `O(n)` 算出最多造成多少伤害；
    - 总回合数固定情况下，每一回合平A的收益（ `f[i]` ）和下毒的收益（ `(limit - i) * g[i]` ）都已知，每回合选收益大的即可。



## 052 单调栈-上 Monotonic Stack

- 经典用法
  - 给定数组每个位置，都求当前位置 **左/右侧 比当前位置 小/大，且 距离最近** 的位置
    - 找小的则严格单调递增
    - 找大的则严格单调递减
  - 数组中 有/无 重复元素 均可
  - 所有调整的总代价为 `O(n)`，单次操作均摊代价为 `O(1)`
- 流程（举例求解最近小邻居）
  - 栈里存下标
  - 栈底到栈顶对应原数组的元素**严格单调递增**，即**大压小**
  - 从左到右遍历原数组，**新元素来了，不违反“大压小”就进**，进不了就弹出元素并 *结算*，遍历完后依次弹出栈里剩余元素并 *结算*
    - 空栈或大于栈顶就入栈，否则不停弹出直到空栈或大于栈顶
  - 如果栈顶被**弹出**，则对栈顶进行 *结算*：
    - 左侧最近的**小于**栈顶的：当初压着的位置（没有则不存在）
    - 右侧最近的**小于**栈顶的：谁让我出来的谁就是（没有则不存在）
    - 如有**重复元素**，则出栈时先记下那个相同元素，栈清空之后再反向遍历答案数组，更新右侧数据
      - 左侧天然是对的
- 性能考虑：用 `std::vector` 代替 `std::stack`，注意要 `reserve`（`std::stack` 没有 `reserve` 方法）
```c++
std::vector<int> stk;
stk.reserve(n);

// 遍历阶段
for (int i = 0; i < n; ++i)
{
    while (!stk.empty() && arr[i] <= arr[stk.back()])
    {
        int cur = stk.back();
        stk.pop_back();
        leftNearestLess[cur] = stk.empty() ? -1 : stk.back();
        rightNearestLess[cur] = i;
    }

    stk.emplace_back(i);
}

// 清算阶段
while (!stk.empty())
{
    int cur = stk.back();
    stk.pop_back();
    leftNearestLess[cur] = stk.empty() ? -1 : stk.back();
    rightNearestLess[cur] = -1;
}

// 修正阶段
// 左侧的答案不需要修正一定是正确的，只有右侧答案需要修正
// 从右往左修正，n-1位置的右侧答案一定是-1，不需要修正
for (int i = n - 2; 0 <= i; --i)
{
    if (rightNearestLess[i] != -1 && arr[rightNearestLess[i]] == arr[i])
    {
        rightNearestLess[i] = rightNearestLess[rightMearestLess[i]];
    }
}
```
- [739. Daily Temperatures](https://leetcode.com/problems/daily-temperatures/)
- [907. Sum of Subarray Minimums](https://leetcode.com/problems/sum-of-subarray-minimums/)
  - Number of subarrays of array `[ll...rr]` containing element `ll <= cur <= rr`: `(cur - ll) * (rr - cur)`. 
- [84. Largest Rectangle in Histogram](https://leetcode.com/problems/largest-rectangle-in-histogram/)
- [85. Maximal Rectangle](https://leetcode.com/problems/maximal-rectangle/)
  - 枚举行，每一行都是一道84题，行与行之间 `h` 的更新直接滚动数组 `O(cols)` 实现之。



## 053 单调栈-下 Monotonic Stack

- 单调栈除经典用法之外，还可以 **维持求解答案的可能性**
  - 发现题目求解内容的 **单调性**（遍历过程中 **单调后继可以排除前驱作为答案的可能**），然后用单调栈来实现
   - 单调栈里的所有对象按照 规定好的单调性来组织
   - 当某个对象进入单调栈时，会从 栈顶开始 依次淘汰单调栈里 **对后续求解没有帮助** 的对象
   - 每个对象从栈顶弹出时 **结算当前对象参与的答案**，随后这个对象 不再参与后续求解答案的过程
   - **入栈和出栈不一定要在一次遍历中同时发生**，可以遍历两次，第一次只进，第二次只出
- [962. Maximum Width Ramp](https://leetcode.com/problems/maximum-width-ramp/)
  - 单调性：若 `a[i] <= a[j]`，`i < j`，则 `a[j]` 能形成的坡 `a[i]` 也都能形成，而且长度更优
  - 严格单调递减栈，第一遍从左到右，只入栈（左端点），第二遍从右到左，枚举右端点，找最优左端点
- [316. Remove Duplicate Letters](https://leetcode.com/problems/remove-duplicate-letters/)
  - 单调性：若 `s[i] > s[j]`，`i < j`，且后面还有 `s[i]`，则当前位置删除 `s[i]` 保留 `s[j]` 一定合法且字典序更小
- [2289. Steps to Make Array Non-decreasing](https://leetcode.com/problems/steps-to-make-array-non-decreasing/)
  - 单调性：从右往左遍历，如果后继能吃前驱，则后继（或者吃了这个后继的更大的鱼接班后）需要`max(turn(succ), 1 + turn(pred))`回合吃完所有右边的鱼
- [1504. Count Submatrices With All Ones](https://leetcode.com/problems/count-submatrices-with-all-ones/)
  - LC第85题“Maximal Rectangle”（上一章最后一题）的变种



## 054 单调队列-上 Monotonic Deque

- 经典用法
  - 滑动窗口在滑动时，想随时得到 **当前滑动窗口** 的 **最大值** 和 **最小值**
  - 窗口滑动的过程中，单调队列所有调整的总代价为 `O(n)`，单次操作的均摊代价为 `O(1)`
  - 单调队列只负责维护最值，窗口本身依旧靠原数组内的双指针 `ll` 和 `rr` 表示
- 流程（举例维护最大值）
  - 双端队列，存下标，从头到尾 **大压小**（严格单调递减），最大值下标为 `deq.front()`
  - 窗口右扩：从尾部入队，如有违反大压小的元素依次从尾部弹出
  - 窗口左缩：看队头下标是否过期，如有过期则从头部弹出
- 性能考虑：用静态数组和左闭右开下标表示双端队列，`std::deque` 常数还是太大
```c++
std::vector<int> arr;
auto n = static_cast<const int>(arr.size());

std::vector<int> minDeq(n), maxDeq(n);
int minDeqL = 0, minDeqR = 0, maxDeqL = 0, maxDeqR = 0;

// 滑动窗口扩张，新加入 `arr[rr]`
auto push = [&arr, &minDeq, &minDeqL, &minDeqR, &maxDeq, &maxDeqL, &maxDeqR](int rr)
{
    while (minDeqL < minDeqR && arr[rr] <= arr[minDeq[minDeqR - 1]])
    {
        --minDeqR;
    }

    minDeq[minDeqR++] = rr;

    while (maxDeqL < maxDeqR && arr[maxDeq[maxDeqR]] <= arr[rr])
    {
        --maxDeqR;
    }

    maxDeq[maxDeqR++] = rr;
};

// 滑动窗口缩小，移除 `arr[ll]`
auto pop = [&arr, &minDeq, &minDeqL, &minDeqR, &maxDeq, &maxDeqL, &maxDeqR](int ll)
{
    while (minDeqL < minDeqR && minDeq[minDeqL] <= ll)
    {
        ++minDeqL;
    }

    while (maxDeqL < maxDeqR && maxDeq[maxDeqL] <= ll)
    {
        ++maxDeqL;
    }
};
```
- [239. Sliding Window Maximum](https://leetcode.com/problems/sliding-window-maximum/)
- [1438. Longest Continuous Subarray With Absolute Diff Less Than or Equal to Limit](https://leetcode.com/problems/longest-continuous-subarray-with-absolute-diff-less-than-or-equal-to-limit/)
  - 问最长窗口：外围右移 `ll`，内部固定 `ll` 右移 `rr`，反过来不行
  - 问最小窗口：外围右移 `rr`，内部固定 `rr` 右移 `ll`
- [P2698 [USACO12MAR] Flowerpot S](https://www.luogu.com.cn/problem/P2698)
  - 求最值之差大于等于 `d` 的最小窗口，外围右移 `rr`，内部固定 `rr` 右移 `ll`



## 055 单调队列-下 Monotonic Deque

- 单调队列可以 **维持求解答案的可能性**
  - 发现题目求解内容的 **单调性**（遍历过程中 **单调后继可以排除前驱作为答案的可能**），然后用单调队列来实现
    - 单调队列里的所有对象按照 规定好的单调性来组织
    - 当某个对象进入单调队列时，会从 对队头开始 依次淘汰单调队列里 **对后续求解没有帮助** 的对象
    - 每个对象从队头弹出时 **结算当前对象参与的答案**，随后这个对象 不再参与后续求解答案的过程
    - **入队和出队不一定要在一次遍历中同时发生**，可以遍历两次，第一次只进，第二次只出
- [862. Shortest Subarray with Sum at Least K](https://leetcode.com/problems/shortest-subarray-with-sum-at-least-k/)
  - 子数组，求和 => 前缀和 + 滑动窗口
  - 记前缀和为 `ps`，以 `arr[rr]` 结尾的子数组，最短要以 `ll` 开始，`k <= ps[rr + 1] - ps[ll]`。
  - **对每一元素找最近的小于等于 `?` 的前缀**，但**不要求每个元素都求出真实值**，只要找出最短即可。
    - 单调队列，小压大，队尾入队前缀和
      - 当不满足小压大，队尾不断出队，直到当前前缀和从队尾入队不违反小压大
        - 要找最近的小于啥啥的元素，队尾一定比大于自己的前驱更优
      - 队头不断出队，直到只有队头 `> ?`
        - `while` 内部更新 `ans`
        - 之前出队的位置，既使和 `rr` 之后的位置构成答案，也不会比当前 `[?...rr]` 这个子数组更短，可以淘汰！
- [1499. Max Value of Equation](https://leetcode.com/problems/max-value-of-equation/)
  - Ask for ax `y1 - x1 + y2 + x2` s.t. `0 < x2 - x1 <= k`. 
  - Find max `y1 - x1` in prefix window `[..)` of `(x2, y2)` of size `k`. 
  - Two ways to do this:
    - (1) Monotonic deque, maintain max of `y - x` in predix window `[..)` of size `k`. 
    - (2) Max heap with lazy removal. 
- [2071. Maximum Number of Tasks You Can Assign](https://leetcode.com/problems/maximum-number-of-tasks-you-can-assign/)
  - Bin Search Ans AND Monotonic Deque
  - Assign top-x skillful workers with one out of top-x eaisest tasks
    - Take easiest task without taking a pill; or
    - If must take a pill, take hardest task doable with the pill. 
    - Maintained by (non-strict) monotonic (ascending) queue (does not need a deque). 
  - Could finish x tasks iff. number of pilled needed `<=` pills we have. 



## 056 并查集-上

```c++
class UnionFind
{
public:
    explicit UnionFind(int size) : root(size), rank(size, 0)
    {
        std::iota(root.begin(), root.end(), 0);
    }

    int find(int x)
    {
        if (root[x] == x) return x;
        return root[x] = find(root[x]);
    }

    void unite(int x, int y)
    {
        int rx = find(x);
        int ry = find(y); 
        if (rx == ry) return;

        if (rank[rx] < rank[ry]) root[rx] = ry;
        else if (rank[ry] < rank[rx]) root[ry] = rx;
        else root[ry] = rx, ++rank[rx];
    }

private:
    int size = 0;
    std::vector<int> root;
    std::vector<int> rank;
};
```
- [P3367 【模板】并查集](https://www.luogu.com.cn/problem/P3367)
- [0839. Similar String Groups](https://leetcode.com/problems/similar-string-groups/)
  - Brute Force Pair AND UnionFind



## 057 并查集-下

- [947. Most Stones Removed with Same Row or Column](https://leetcode.com/problems/most-stones-removed-with-same-row-or-column/)
  - 同行，同列都算作一个组
  - 同组内石头一定能消到只剩一个
- [2092. Find All People With Secret](https://leetcode.com/problems/find-all-people-with-secret/)
  - **一次合并发生后，涉及节点只要没有作为中继合并其它组，这次合并就可以直接重设这次合并涉及的结点的根来撤销**
    - 这不算可撤销并查集
- [2421. Number of Good Paths](https://leetcode.com/problems/number-of-good-paths/)
  - UnionFind AND max(val(edge)) sorting
- [928. Minimize Malware Spread II](https://leetcode.com/problems/minimize-malware-spread-ii/description/)
  - Connected Component excluding virus nodes savable iff only one virus infects it



## 058 洪水填充

- [200. Number of Islands](https://leetcode.com/problems/number-of-islands/)
  - Aka DFS. 
- [130. Surrounded Regions](https://leetcode.com/problems/surrounded-regions/)
  - Flood fill all boundary `'O'`s into `'F'`s, then loop all elements and modify.  
- [827. Making A Large Island](https://leetcode.com/problems/making-a-large-island/)
  - DFS Mark Island Group and Brute Force
- [803. Bricks Falling When Hit](https://leetcode.com/problems/bricks-falling-when-hit/)
  - 时光倒流



## 059 建图、链式前向星、拓扑排序

- Graph:
  - `n` vertices; 
  - `m` edges. 
- 邻接矩阵 Adjacency Matrix
  - `O(n**2)` space, at most 2000 vertices. 
  - `std::vector<std::vector<int>> am;`
- 邻接表 Adjacency List
  - `O(nm)` space (if requires static memory preallocation)
  - `std::vector<std::vector<std::pair>> al;`
  - `al[source]` stores all edges originating from vertex `source`, in pair `{target, weight}`.  
- 链式前向星 (Static Adjacency List, "Forward-Star List")
  - `O(n + m)` space (even when static mememory preallocation)
  - **1-indexed!**
  - E.g., Edges added in order `#1 (1 -> 2)`, `#2 (1 -> 3)`
    - `head == {0, 2, 0, 0}`
    - `next == {0, 0, 1}`
    - `to == {0, 2, 3}`
    - `cnt == 3` (`cnt == 1` when graph is empty.)
```c++
constexpr int kMaxVerts = 110;  // 点的最大数量
constexpr int kMaxEdges = 210;  // 边的最大数量

// Vertices and edges are all 1-indexed!
using VertIdx = int;
using EdgeIdx = int;
using Weight = int;

// Size (num of edges) of the current graph. 
EdgeIdx cnt = 0;

// Vertex Property. 
// Edge ID of the most-recently-added edge originating from this vertex. 
std::array<EdgeIdx, kMaxVerts> head = {0}; 

// Edge Property. 
// Edge ID of the previously-added edge originating from the same source vertex. 
std::array<EdgeIdx, kMaxEdges> next = {0};

// Edge Property. 
// Vertex ID of target vertex of this edge. 
std::array<VertIdx, kMaxEdges> to = {0};

// Edge Property. 
// Weight of this edge if this graph is weighted. 
std::array<Weight, kMaxEdges> weight = {0};

// Totally n vertices, indexed from 1 to n. 
void build(int n)
{
    cnt = 0;
    std::fill(head + 1, head + n + 1, 0);
}

// Edge (s -> t), weight w. 
void addEdge(VertIdx s, VertIdx t, Weight w)
{
    next[++cnt] = head[s];
    head[s] = cnt;
    to[cnt] = t;
    weight[cnt] = w;
}

void traverse(int n)
{
    for (int i = 1; i <= n; ++i)
    {
        std::cout << i << " (neighbor, weight): ";

        for (EdgeIdx e = head[i]; 0 < e; e = next[ei])
        {
            std::cout << "( " << to[e] << ", " << weight[e] << " ) ";
        }
    }

    std::cout << '\n';
}
```
- Topological Sort (with Minimal Dict Order)
```c++
const int n = numVertices;
const int m = numEdges;

// Vertices and edges are all 1-indexed!
using VertIdx = int;
using EdgeIdx = int;
std::vector<EdgeIdx> head(n + 1, 0);
std::vector<EdgeIdx> next(m + 1, 0);
std::vector<VertIdx> to(m + 1, 0);
EdgeIdx cnt = 0;

// Build graph, vertices are 1-indexed. 
for (auto [s, t] : edges)
{
    next[++cnt] = head[s];
    head[s] = cnt;
    to[cnt] = t;
}

std::vector<VertIdx> ans;
ans.reserve(n);

std::vector<VertIdx> inDegree(n + 1, 0);

for (VertIdx s = 1; s <= n; ++s)
{
    for (EdgeIdx e = head[s]; 0 < e; e = next[e])
    {
        ++inDegree[to[e]];
    }
}

// Use min heap to output topological sort and vert ids ascending. 
// Could use regular queue if not requiring vert ids ascending. 
std::priority_queue<VertIdx, std::vector<VertIdx>, std::greater<VertIdx>> heap;

for (VertIdx i = 1; i <= n; ++i)
{
    if (!inDegree[i])
    {
        heap.emplace(i);
    }
}

while (!heap.empty())
{
    VertIdx curr = heap.top();
    heap.pop();
    ans.emplace_back(curr);  // 1-indexed!

    for (EdgeIdx e = head[curr]; 0 < e; e = next[e])
    {
        if (!--inDegree[to[e]])
        {
            heap.emplace(to[e]);
        }
    }
}
```
- [210. Course Schedule II](https://leetcode.com/problems/course-schedule-ii/)
  - Topological sort examplar. 
- [269. Alien Dictionary](https://leetcode.com/problems/alien-dictionary/)
  - Build graph with consecutive word pairs, topological sort. 
- [936. Stamping The Sequence](https://leetcode.com/problems/stamping-the-sequence/)
  - Topological sort manner
  - Bipartie graph `{target indices} -> {stampable indices}`. 
    - `g[i] = {a, b, c, ...}` indicates that 
    - `target[i]` could be invalidated by stamping at positions `a, b, c, ...`



## 060 拓扑排序的扩展技巧

- 利用拓扑排序的过程，**上游节点逐渐推送消息给下游节点**
  - 本质上属于树形DP了
- [P4017 最大食物链计数](https://www.luogu.com.cn/problem/P4017)
  - 推送消息：
- [851. Loud and Rich](https://leetcode.com/problems/loud-and-rich/)
- [2050. Parallel Courses III](https://leetcode.com/problems/parallel-courses-iii/)
  - 推送消息：自己所在链，从头一直上到、上完自己所需时间
- [2127. Maximum Employees to Be Invited to a Meeting](https://leetcode.com/problems/maximum-employees-to-be-invited-to-a-meeting/)
  - **拓扑排序之后，只有成环的结点才会有非零入度**
  - 推送消息：不包括自己的最长链长度
    - 答案有两种可能
      - 每一个小环（两个节点组成的小环 `(s, t)`），都可以坐下 `2 + deep[s] + deep[t]` 个人
      - 多个节点组成的大环，只要最大的，坐下大环大小个人



## 061 最小生成树

- 最小生成树：
  - 在 无向带权连通图 中选择一些边，保证连通性的情况下，边的总权值最小
  - 最小生成树 不一定唯一
  - 最小生成树 一定是 **最小瓶颈树**
- 这一票贪心算法为什么是对的？：因为每一步得到的都是某一棵最小生成树的子集。
  - 概念表记
    - *当前边集*：某一棵最小生成树的边集的子集；
    - *切割*：把这张图的节点分成两个集合；
    - *切割的最轻量级边*：横跨这个切割（源和汇分列于切割两边）的边中，权值最小的；
    - *切割尊重当前边集*：当前边集中的边一律不横跨这个切割。
  - 尊重当前边集的切割的最轻量级边，加入当前边集后，当前边集依旧是某棵最小生成树边集的子集。
    - 这条最轻量级边记为 `e = (u, v)`，当前边集背后的最小生成树记为 `T`；
    - 假设 `T` 不包含 `e`，则将 `e` 加入 `T` 之后，会形成一个环；
    - 这个环中至少还有一条边 `e' = (x, y)` 横跨这个切割（不然这个最小生成树直接按分割裂成两边了，不连通了）；
    - 用 `e` 替换 `T` 中的 `e'`，得到的新树 `T'` 仍旧连通无环且总权重不大于 `T`，即，`T'` 也是最小生成树。
- **Kruskal 算法**（**最常用**）  
  - 起源于全图最便宜的边
    - 不用建图，存一个边 `(s, t, w)` 的数组；
    - 所有边从小到大排序，依次考虑每条边，若加入不成环则加入；
      - 成环判定：并查集，如果源和汇已经属于同一连通域，则加入这条边会成环。
    - 遍历完了为止，看边数是不是 `n - 1`。
  - 复杂度：`O(n + m log m)`，适用于稀疏图。
- *Prim 算法*（大厂笔试里属于废物算法）
  - 起源于随便一个节点
    - 链式前向星建无向图（两倍边数），一个小根堆存边（按权值组织），一个哈希表存已访问节点；
    - 访问 `1` 号结点，将起源于 `1` 号节点的边全部放进小根堆；
    - 不断加入最便宜的且汇未访问的边，同时把起源于这个汇的边也都放进小根堆；
    - 堆空了为止，看节点数是不是 `n`。
  - 复杂度：
    - 上述原始版本：`O(n + m log m)`，纯纯虫豸；
    - 优化：`O((n + m) log n)`，除非是及其稠密的图，不然不会有性能提升。
  - 优化版本：
    - **反向索引堆**：
      - 一个小根堆存 `(结点, 到达节点的花费)`（按花费组织）
      - 一个索引存结点（三种可能：已访问过；未访问过但已在堆里；未访问过且从未进过堆）；
    - 加入一条边时，如果汇已经在堆内，则**更新堆内的已有项**，而不是再塞一个新项；
    - 操作堆时同步更新索引。
    - 看 064 节 Dijkstra 算法处的详细内容。
- [P3366 【模板】最小生成树](https://www.luogu.com.cn/problem/P3366)
- [1168. Optimize Water Distribution in a Village](https://leetcode.com/problems/optimize-water-distribution-in-a-village/)
  - Dummy node to all houses, edges with weights `wells[i - 1]`
- [1697. Checking Existence of Edge Length Limited Paths](https://leetcode.com/problems/checking-existence-of-edge-length-limited-paths/)
  - UnionFind unite edges less equal to limit of query (sorted w.r.t. limit).
- [P2330 [SCOI2005] 繁忙的都市](https://www.luogu.com.cn/problem/P2330)
  - 最小生成树 一定是 **最小瓶颈树**



## 062 宽度优先遍历及其扩展

- BFS
  - 逐层扩散，从源头点到目标点扩散了几层，最短路就是多少；
  - 使用条件：**任意两节点之间距离相同（无向图）**；
  - BFS 开始时，可以是 **单个源头** 或 **多个源头**；
  - BFS 队列可以 **单个弹出** 或 **整层弹出**；
    - 配合整层弹出，到达新一层时队列大小就是这一层的节点数目；
  - 结点 **入队时** 须 **标记状态**，防止同一节点重复入队；
    - 如果不在入队时标记而是访问（出队）时标记，会导致 **同一节点重复入队**
    - 可以采用一个队列加一个哈希表，或者干脆两个哈希表（详见下面逐层标记已访问节点的内容）
  - 可能包含 *剪枝*
    - Dijkstra
    - A*
    - ...
  - 骚：
    - **逐层 BFS**
      - 例：[1162. As Far from Land as Possible](https://leetcode.com/problems/as-far-from-land-as-possible/)
      - 骚上加骚：**逐层标记已访问节点**
        - 一个节点可以被访问多次，因此不能入队即标记，但又不能搞成死循环
          - 所以搞 **两个哈希表表示当前层内容和下一层内容** 
          - **每进入新一层时，统一标记本层节点为已访问**
          - 每层结束时，`currLevel = std::move(nextLevel);`
        - 例：[126. Word Ladder II](https://leetcode.com/problems/word-ladder-ii/)
    - **多起点 BFS**
      - 例：[126. Word Ladder II](https://leetcode.com/problems/word-ladder-ii/)
    - **0-1 BFS（双端队列）**
      - 例：[2290. Minimum Obstacle Removal to Reach Corner](https://leetcode.com/problems/minimum-obstacle-removal-to-reach-corner/)
    - **优先队列 BFS**
      - 例：[407. Trapping Rain Water II](https://leetcode.com/problems/trapping-rain-water-ii/)
- 0-1 BFS
  - 图 `G = {V, E}` 中所有边的权重 **只有 0 和 1 两种值**，求源点到目标点的最短距离;
    - 如果指定目标点，则只更新到目标点的距离；
    - 如果不指定目标点，则更新整张图。
  - 时间复杂度：`O(|V| + |E|)`。
  - 流程：
    - `dist[i]` 表示源点到 `i` 点的最短距离，初始化为正无穷；
    - 源点 进入 双端队列，`dist[source] = 0;`；
    - 双端队列 头部弹出 `x`：
      - 如果 `x` 是目标点，返回 `dist[x]`；
      - 考察从 `x` 出发的每一条边 `(x, y)`，权重为 `w`：
        - 当 `dist[xr] + w < dist[y]` 时处理此边，否则忽略之；
        - 更新 `dist[y] = dist[x] + w;`；
        - 如果 `w == 0`，从 **头部** 入队；
        - 反之，`w == 1`，从 **尾部** 入队。
  - FAQ
    - 为什么不能普通BFS？
      - 因为边权重不同；
    - 为什么不需要 `visited` 标记节点？
      - 有重复节点就有了，`dist` 严格单调递减，自动修正；
    - 正确性？
      - 队列里所有节点到源点到距离差不超过 `1`；
      - 两个方向入队，实现近的一定比远的先访问。
        - 注意队列里面可能有重复节点的（一个节点最多入队两次，也最多弹出两次）；
        - 记录的距离比真实值大的节点有被 重新修正 的机会。
        - 最多被修正几次？一次，因为这是 0/1 图，队内节点距离最多差 1。
```c++
std::deque<VertIdx> deq;
deq.emplace_front(s);

std::vector<Weight> dist(n, std::numeric_limits<int>::max());
dist[s] = 0;

while (!deq.empty())
{
    VertIdx x = deq.front();
    deq.pop_front();

    if (x == target)
    {
        return dist[x];
    }

    for (EdgeIdx e = head[s]; e != 0; e = next[e])
    {
        VertIdx y = to[e];

        if (dist[y] <= dist[x] + weight[e])
        {
            continue;
        }

        dist[y] = dist[x] + weight[e];

        if (weight[e] == 0)
        {
            deq.emplace_front(y);
        }
        else  // weight[e] == 1
        {
            deq.emplace_back(y);
        }
    }
}
```
- [1162. As Far from Land as Possible](https://leetcode.com/problems/as-far-from-land-as-possible/)
  - Multiple starting points, Levelwise BFS, Mark visited when enqueuing. 
- [691. Stickers to Spell Word](https://leetcode.com/problems/stickers-to-spell-word/)
  - Status bin mask DP (indices settled in target), or
  - BFS (full bin mask to zero)
- [2290. Minimum Obstacle Removal to Reach Corner](https://leetcode.com/problems/minimum-obstacle-removal-to-reach-corner/)
  - Model the grid as a graph where cells are nodes and edges are between adjacent cells. 
    - Edges to cells with obstacles have a cost of 1 and all other edges have a cost of 0.
  - 0-1 BFS, or
  - Dijkstra (sub-optimal)
- [1368. Minimum Cost to Make at Least One Valid Path in a Grid](https://leetcode.com/problems/minimum-cost-to-make-at-least-one-valid-path-in-a-grid/)
  - 0-1 BFS
- [407. Trapping Rain Water II](https://leetcode.com/problems/trapping-rain-water-ii/)
  - BFS with min heap
- [126. Word Ladder II](https://leetcode.com/problems/word-ladder-ii/)
  - BFS, record inverse graph, then DFS for plan;
  - TLE if not using inverse graph, too many initial choices;
  - Use two hash sets instead of a queue (and visited hash set).
    - Levelwise markings of visited nodes. 



## 063 双向广搜 Bidirectional BFS

- 用途1：
  - 剪枝优化，哪侧数量少就从哪侧展开；
- **用途2（重要，本体）**：
  - 全量样本 不允许递归完全展开，但是 半量样本 可以完全展开。举例：
  - 长度 `40` 的数组，暴力枚举每个元素要或者不要，复杂度为 `2**40 ~ 10**12`；
  - 分别枚举前一半和后一半，最后线性整合答案，复杂度为 `2 * (2 ** 20) + (2 ** 20) ~ 10**6`。
```c++
std::array<int> arr;
auto n = static_cast<const int>(arr.size());

int lsSize = 0;
int rsSize = 0;

std::array<int, 1 << 20> ls;
std::array<int, 1 << 20> rs;

void f(int b, int e, int cur, int * res, int * offset)
{
    if (b == e)
    {
        res[(*offset)++] = cur;
    }
    else
    {
        f(b + 1, e, cur, res, offset);           // Drop arr[b].
        f(b + 1, e, cur + arr[b], res, offset);  // Take arr[b].
    }
}

f(0, n >> 1, 0, ls.data(), &lsSize);
f(n >> 1, n, 0, rs.data(), &rsSize);

std::sort(ls.begin(), ls.begin() + lsSize);
std::sort(rs.begin(), rs.begin() + rsSize);
```
- [P4799 [CEOI2015 Day2] 世界冰球锦标赛](https://www.luogu.com.cn/problem/P4799)
- [1755. Closest Subsequence Sum](https://leetcode.com/problems/closest-subsequence-sum/)
  - BiDir BFS AND 2Sum Closest



## 064 Dijkstra 算法、分层图最短路

- Dijkstra 算法
  - 单源最短路算法——给定一个源点，求源点到每个点的最短路径长度；
  - 要求**有向图**，且**边的权值没有负数**。
  - 流程：
    - `dist` 数组记录当前各节点的距离，初始化为正无穷；
    - `visited` 数组记录每个节点是否已被扩展过（扩展这个节点出发的边，每个节点只会被扩展一次）；
    - 小根堆记录节点编号**以及距离**；
      - 一定要记录距离，依靠堆内记录的距离排序，**不能**只记录节点编号！
      - `dist` 是会变的，**如果比较器实时依赖 `dist`，堆会不合法**！
      - 想玩骚的，折腾下面那个反向索引堆去。
    - 每次扩展堆内距离最小的且之前没扩展过的节点，对于这个节点有向连接至的汇节点：
      - 普通版：
        - 这个汇节点之前扩展过，或者不能让其他没弹出节点距离变小，就忽略；
        - 这个汇节点加入堆，更新变小的距离。
      - 进阶版：
        - 反向索引堆；
        - 每次扩展距离最小的节点，但不直接入堆，而是依靠索引更新堆和距离（`heapUpdate`）。
      - 进阶版和普通版的区别：
        - 每次从堆里拿出一个最小的节点进行拓展时，这个节点：
          - 是全新的：一样；
          - 已经在堆里了但还没拓展过：普通版会塞一个重复的节点（但新塞的小的节点会先被拓展到），进阶版更新堆内那个已有节点；
          - 已经被拓展过了：普通版拓展边之前会因为 `visited` 而直接跳过，进阶版压根不会有这种节点（因为堆内没有重复节点）。
    - 常数优化：**如果堆里拿出的节点就是目标点，则直接终止算法**
```c++
void dijkstra(int source)
{
    std::vector<int> dist(n + 1, std::numeric_limits<int>::max());
    dist[source] = 0;

    std::vector<std::uint_8> visited(n + 1, false);

    std::priority_queue<
            std::pair<int, int>, 
            std::vector<std::pair<int, int>>, 
            std::greater<>
    > minHeap;

    minHeap.emplace(0, source);

    while (!minHeap.empty())
    {
        auto [d, s] = minHeap.top();
        minHeap.pop();

        if (visited[s])
        {
            continue;
        }

        visited[s] = true;

        for (auto [t, w] : adjacencyList[s])
        {
            if (d + w < dist[t])
            {
                dist[t] = d + w;
                minHeap.emplace(dist[t], t);
            }
        }
    }
}
```
- **反向索引堆** (Inverse Indexed Heap)
  - 一个 `where` 数组查询一个 `Key` 是从未进过堆（`-1`），在堆里（`>=0`），还是已经被弹出了（`-2`）；
  - 手写一个 `heapSwap` 函数来交换两个两个下标（交换堆数组，同时更新 `where` 数组）；
  - 手写堆，`heapPush` 和 `heapify`；
  - 公开接口： `heapPop` 和 `heapUpdate`。
  - 使用反向索引堆的 `Dijkstra` **不**再需要 `visited` 记录是否已经拓展过某个节点：
    - 每次拿到新节点都扩展一遍 `heapUpdate` 即可。
      - 反向索引堆内部一定不会有重复节点，
      - 弹出过的节点也一定不会再次进堆，
      - 因此根本不会出现重复扩展的情况。
```c++
// 反向索引表 where，堆 heap，比较索引 dist[heap[i]]
constexpr int kMaxHeapSize = ...;

int heapSize = 0;
std::array<int, kMaxHeapSize> heap = {};
std::array<int, kMaxHeapSize> dist = {};
std::array<int, kMaxHeapSize> where = {};

void clear()
{
    heapSize = 0;
    std::fill(dist.begin(), dist.end(), std::numeric_limits<int>::max());
    std::fill(where.begin(), where.end(), -1);
}

void heapSwap(int i, int j)
{
    // 快照: heap[i] == h1, heap[j] == h2
    std::swap(heap[i], heap[j]);

    // 现在有 heap[i] == h2, heap[j] == h1
    // 更新索引为 where[h2] = i, where[h1] = j
    where[heap[i]] = i;
    where[heap[j]] = j;
}

void heapPush(int i)
{
    // 注意这里【不能】用 (i - 1) >> 1，移位的话 i == 0 时会溢出！
    while (dist[heap[i]] < dist[heap[(i - 1) / 2]])
    {
        heapSwap(i, (i - 1) / 2);
        i = (i - 1) / 2;
    }
}

void heapify(int i)
{
    for (int l = (i << 1) + 1; l < heapSize; )
    {
        int best = (l + 1 < heapSize && dist[heap[l + 1]] < dist[heap[l]]) ? l + 1 : l;
        best = dist[heap[i]] < dist[heap[best]] ? i : best;

        if (best == i)
        {
            break;
        }

        heapSwap(i, best);
        i = best;
        l = (i << 1) + 1;
    }
}

void heapPop()
{
    heapSwap(0, --heapSize);

    // 注意【索引必须在 swap 之后更新】。
    // 因为 swap 对 where 并不是真正的交换，而是更新实际位置；
    // 也就是说如果 swap 之前 where 是 -1 或者 -2 的话，
    // swap 之后会被覆盖成真实下标（ >= 0 ）
    where[heap[heapSize]] = -2;
    heapify(0);
}

void heapUpdate(int v, int d)
{
    if (where[v] == -1)
    {
        // Insert
        dist[v] = d;
        heap[heapSize] = v;
        where[v] = heapSize++;
        heapPush(where[v]);
    }
    else if (0 <= where[v])
    {
        // Update
        dist[v] = std::min(dist[v], d);
        heapPush(where[v]);
    }
    // else Ignore
}

void dijkstra(int source)
{
    heapUpdate(source, 0);

    while (0 < heapSize)
    {
        int s = heap[0];
        heapPop();

        for (int e = head[s]; e != 0; e = next[e])
        {
            int t = to[e];
            int w = weight[e];
            heapUpdate(t, dist[s] + w);
        }
    }
}
```
- **骚**操作（详见下面例题）：
  - **最大边权最短路**
    - 最短路也可以定义为路径上所有边权的最大值，Dijkstra 算法同样适用
    - 每次 `tDist` 更新为 `max(sDist, w)` 而不是 `sDist + w`
  - **分层图最短路**
    - 节点带状态
    - 相当于把平面图上下堆叠几层，相邻层的同号节点之间有边连接
      - 例如，本地充一格电，或者用一张免费机票，等等
    - [图例](https://www.luogu.com.cn/article/ul9rz6oi)
- [743. Network Delay Time](https://leetcode.com/problems/network-delay-time/)
- [P4779 【模板】单源最短路径（标准版）](https://www.luogu.com.cn/problem/P4779)
- [1631. Path With Minimum Effort](https://leetcode.com/problems/path-with-minimum-effort/)
  - Dijkstra Mk1 max-so-far as dist. 
  - **最短路也可以定义为路径上所有边权的最大值，Dijkstra 算法同样适用**
- [778. Swim in Rising Water](https://leetcode.com/problems/swim-in-rising-water/)
  - Dijkstra Mk1 max-so-far as dist, analog of 1631 (above). 
- [864. Shortest Path to Get All Keys](https://leetcode.com/problems/shortest-path-to-get-all-keys/)
  - State-compression key holding status. 
  - Dist BFS **with state**: `dist[x][y][state]`
- [LCP 35. 电动车游城市](https://leetcode.cn/problems/DFPeFJ/)
  - **分层图**：节点带状态的 Dijkstra，节点重新定义为 `(节点, 剩余电量)`
  - 每次扩展为：不充电直接去隔壁，或者当前节点充一格电
    - 不充更多是因为 Dijkstra 只拓展当前最小节点
    - 充多于一格电，相当于再次扩展了一个新节点
  - **USE CONTINUE ONLY IMMEDIATELY INSIDE LOOPS!!!**
- [P4568 [JLOI2011] 飞行路线](https://www.luogu.com.cn/problem/P4568)
  - 还是分层图



## 065 A*, Floyd, Bellman-Ford, And SPFA

- A*
  - 单源最短路算法，和 Dijkstra 有一处不同
    - 堆内排序不只用当前距离，而是当前距离 + 剩余距离估值
    - 估值要 <= 真实值
      - 估值 <= 真实值的情况下，越接近真实值，越快
      - 其余情况，则有负面影响
  - 需要额外信息
    - 例如，均匀正方形网格上的曼哈顿距离，欧几里得距离，对角线距离 `max(dx, dy)`，等等
    - 对于平凡图不适用
- Floyd 算法
  - **任意两点间最短短路问题**
  - 适用于任何图（可以有负边权，只要没有负环）
  - 时间复杂度 `O(n**3)`，空间复杂度 `O(n**2)`
  - `for k for i for j dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j]);`
    - 相当于三维 DP，**中继 `k` 要在最外层，最先被枚举**
    - 只要有一个中继 `k` 的 `dist` 是正确的，就能正确更新
    - 因为 `k` 在最外层，总会有一个已经被正确更新好的，`k` 在其它层则不行
- Bellman-Ford 算法
  - 解决**带负权边**的图上的**单源最短路问题**
    - 复杂度很高，只适用于小图！
    - 没有负边权时要用 Dijkstra
  - **松弛操作（Relaxation）**：
    - 假设源点为 `A`，从 `A` 到任意点 `F` 的最短距离为 `dist[F]`
    - 假设从点 `P` 出发的某条边去往点 `S`，边权为 `W`
    - 如果发现 `dist[P] + W < dist[S]`，即通过这条边中转可以让 `dist[S]` 变小
    - 那么就说，`P` 出发的这条边对点 `S` 进行了 松弛操作
  - 过程：
    - `dist[source] = 0; dist[other] = +oo;`
    - 每一轮考察每条边，每条边都尝试进行松弛操作，那么若干点的 `dist` 会变小
    - 不再有可松弛的点时，结束
  - Dijkstra 每轮扩展一个最小距离的点，Bellman-Ford 每轮挨个松弛每一条边
  - 复杂度
    - 最短路存在的情况下
      - 每一次松弛操作都会使
      - 源点到被松弛点的 “目前已知最短路” 的边数
      - +1 甚至更多
    - 源点到任何一个点的最短路最长都是 `n - 1`，因此松弛轮数最多 `n - 1` 轮
    - 时间复杂度 `O(mn)`，只适用于小图！
  - **推广**：判断某个点出发能不能到达 **负环**
    - 当松弛到第 `n` 轮时仍能接着松弛，则一定存在负环！
- SPFA（Shortest Path Faster Algorithm）
  - 对 Bellman-Ford 的常数优化
    - 复杂度依旧很高，只适用于小图！
    - 没有负边权时要用 Dijkstra
  - 只有上一次被某条边松弛过的节点所连接的边，才有可能引起下一次松弛操作
    - 用队列维护 “这一轮哪些节点的 `dist` 变小了”
    - 下一轮只需要对这些点的所有边考察有无松弛操作即可。
- [P2910 [USACO08OPEN] Clear And Present Danger S](https://www.luogu.com.cn/problem/P2910)
- [787. Cheapest Flights Within K Stops](https://leetcode.cn/problems/cheapest-flights-within-k-stops/)
  - 可以接着 Dijkstra 玩分层图最短路（但是慢！）
  - 也可以 Bellman-Ford，限制每一轮，每个点最多只能被松弛一次。松弛 `k + 1` 轮即可。
    - 每次松弛都 *更新一个拷贝的新表* （而不是自己）就可以避免松弛联动。
- [P3385 【模板】负环](https://www.luogu.com.cn/problem/P3385)
  - SPFA
  - 有**多个测试点**的 ACM 风格题目，**记得重设答案**！



## 066 从递归入手一维动态规划

- DP
  - 知道怎么算 vs **知道怎么试**： DP 解决知道怎么试 的类型；
  - 递归过程中 *反复调用同一个子问题的解* 的递归才有改动态规划的必要。
- 见识过的一维 DP 方案：
  - 下标 DP ：
    - 前缀；
    - 后缀（字符串常用）；
  - 数值 DP ：
    - [322 Coin Change](https://leetcode.com/problems/coin-change/)；
    - [467 Unique Substrings](https://leetcode.com/problems/unique-substrings-in-wraparound-string/)）。
      - 枚举起始字符达到去重效果
    - [940. Distinct Subsequences II](https://leetcode.com/problems/distinct-subsequences-ii/)
      - 枚举结尾字符
    - [1987. Number of Unique Good Subsequences](https://leetcode.com/problems/number-of-unique-good-subsequences/)
      - 枚举首尾字符
- [509. Fibonacci Number](https://leetcode.cn/problems/fibonacci-number/)
- [983. Minimum Cost For Tickets](https://leetcode.com/problems/minimum-cost-for-tickets/)
- [91. Decode Ways](https://leetcode.com/problems/decode-ways/)
- [639. Decode Ways II](https://leetcode.com/problems/decode-ways-ii/description/)
- [264. Ugly Number II](https://leetcode.com/problems/ugly-number-ii/)
  - 给定质因数组成，如何从小到大枚举？
  - 每一个丑数一定都是某个前驱丑数乘2，3或5乘出来的；
  - 假设当前一步生成的新丑数是某个前驱乘2，则下次再乘2的一定是这个前驱的下一位；
  - 特殊注意：当前一步生成的新丑数可能有多个来源，一旦被采用，所有来源都要前进一步。
- [32. Longest Valid Parentheses](https://leetcode.com/problems/longest-valid-parentheses/)
  - 可以前缀DP做
  - 看见括号匹配了，也可以用栈做
  - 还有更骚的，纯骚
- [467. Unique Substrings in Wraparound String](https://leetcode.com/problems/unique-substrings-in-wraparound-string/)
  - 字符数值DP，而不是下标DP。
  - 合法子串之间一定不重合：枚举起始位置，每次看下一位是否能填进来，更新目前连续合法串长度。
  - 以给定字符为结尾的最长子串一定包含所有更短的以这个字符结尾的子串。
- [940. Distinct Subsequences II](https://leetcode.com/problems/distinct-subsequences-ii/)
  - 枚举结尾字符
- [1987. Number of Unique Good Subsequences](https://leetcode.com/problems/number-of-unique-good-subsequences/)
  - 枚举首尾字符


## 067 从递归入手二维动态规划







## 068 见识更多二维动态规划题目







## 069 从递归入手三维动态规划







