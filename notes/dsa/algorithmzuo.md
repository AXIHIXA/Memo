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
// Differs from std::upper_bound, 
// which locates 1st element > num (num < element == true). 
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





## 059 Graph

- Graph with n nodes and m edges. 
- 邻接矩阵 Adjacency Matrix
  - O(n**2) space
  - `std::vector<std::vector<int>> am;`
- 邻接表 Adjacency List
  - O(nm) space (if requires static mem prealloc)
  - `std::vector<std::vector<std::pair>> al;`
  - `al[source]` stores all edges originating from vertex `source`, in pair `{target, weight}`.  
- 链式前向星 (1-indexed!)
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