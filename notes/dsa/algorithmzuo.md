# algorithmzuo


## 000 Routines

- OJ Stuff
```c++
static const int init = []
{
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);
    std::cout.tie(nullptr);
    std::setvbuf(stdin, nullptr, _IOFBF, 1 << 20);
    std::setvbuf(stdout, nullptr, _IOFBF, 1 << 20);
    return 0;
}();
```
- STL Random
```c++
// True random device. Expensive. Non-configurable. 
auto seed = std::random_device()();

// Random number engine, manual seed. 
std::default_random_engine e(seed);

// Uniform distribution. 
std::uniform_int_distribution dd();     // U[0, INT_MAX]
std::uniform_int_distribution d(a, b);  // U[a, b]

// Generate random number. 
int r1 = d(e);                          // Sample U[a, b]
int r2 = d(e, {c, d})                   // Sample U[c, d]

// Two equivalent iota vectors. 
std::vector<int> v(50001);
std::iota(v.begin(), v.end(), 0);
std::generate(v.begin(), v.end(), [n = 0] mutable { return n++; });

// Random shuffle. 
// https://en.cppreference.com/w/cpp/algorithm/random_shuffle
// std::random_shuffle is deprecated. Use std::shuffle. 
std::shuffle(v.begin(), v.end(), e);
```



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
		// 右孩子，l+1
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



## 041最大公约数、同余原理

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


## 044 前缀树原理和代码详解

- [Trie](../../notes/dsa/Trie.cpp)
- [LC 421. Maximum XOR of Two Numbers in an Array](https://leetcode.com/problems/maximum-xor-of-two-numbers-in-an-array/)
- [LC 212. Word Search II](https://leetcode.com/problems/word-search-ii/)



## 059 Graph

- Graph with n nodes and m edges. 
- 邻接矩阵 Adjacency Matrix
  - O(n**2) space
  - `std::vector<std::vector<int>> am;`
- 邻接表 Adjacency List
  - O(nm) space (if requires static mem prealloc)
  - `std::vector<std::vector<std::pair>> al;`
  - `al[source]` stores all edges originating from vertex `source`, in pair `{target, weight}`.  
- 链式前向星 (Static Adjacency List) (1-indexed!)
  - O(n + m) space (even for static mem prealloc)
  - E.g., Edges added in order `#1 (1 -> 2)`, `#2 (1 -> 3)`
    - `head == {0, 2, 0, 0}`
    - `next == {0, 0, 1}`
    - `to == {0, 2, 3}`
    - `cnt == 3`
```c++
constexpr int kMaxM = 21;  // 边的最大数量
constexpr int kMaxN = 11;  // 点的最大数量

// Index: Vertex ID. 
// Value: ID of the most-recently-added edge originating from this vertex. 
int head[kMaxN] {0}; 

// Index: Edge ID. 
// Value: ID of the previously-added edge originating from the same source vertex. 
int next[kMaxM] {0};

// Index: Edge ID. 
// Value: ID of target vertex of this edge. 
int to[kMaxM] {0};

// Index for the next edge to add. 
int cnt {1};

// 如果边有权重，那么需要这个数组
int weight[kMaxM] {0};

// Totally n vertices, indexed from 1 to n. 
void build(int n)
{
    cnt = 1;
    std::fill(head + 1, head + n + 1, 0);
}

// Edge (s -> t), weight w. 
void addEdge(int s, int t, int w)
{
    next[cnt] = head[s];
    to[cnt] = t;
    weight[cnt] = w;
    head[s] = cnt++;
}

// Add two directed edges for undirected graphs. 

void traverse(int n)
{
    for (int i = 1; i <= n; ++i)
    {
        std::cout << i << " (neighbor, weight): ";

        for (int ei = head[i]; 0 < ei; ei = next[ei])
        {
            std::cout << "( " << to[ei] << ", " << weight[ei] << " ) ";
        }
    }

    std::cout << '\n';
}
```
- Topological Sort (with Minimal Dict Order)
```c++
int n = numVertices;
int m = numEdges;

// Vertices and edges are all 1-indexed!
std::vector<int> head(n + 1, 0);
std::vector<int> next(m + 1, 0);
std::vector<int> to(m + 1, 0);
int cnt = 1;

// Build graph, vertices are 1-indexed. 
for (auto [s, t] : edges)
{
    next[cnt] = head[s];
    to[cnt] = t;
    head[s] = cnt++;
}

std::vector<int> ans;
ans.reserve(n);

std::vector<int> inDegree(n + 1, 0);

for (int s = 1; s <= n; ++s)
{
    for (int e = head[s]; 0 < e; e = next[e])
    {
        ++inDegree[to[e]];
    }
}

std::priority_queue<int, std::vector<int>, std::greater<int>> heap;

for (int i = 1; i <= n; ++i)
{
    if (!inDegree[i])
    {
        heap.push(i);
    }
}

while (!heap.empty())
{
    int curr = heap.top();
    heap.pop();
    ans.emplace_back(curr);  // 1-indexed!

    for (int e = head[curr]; 0 < e; e = next[e])
    {
        if (!--inDegree[to[e]])
        {
            heap.push(to[e]);
        }
    }
}
```
- [LC 269 Alien Dictionary](https://leetcode.com/problems/alien-dictionary/)
  - Build graph with consecutive word pairs, topological sort. 
- 1