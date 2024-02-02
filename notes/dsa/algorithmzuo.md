# algorithmzuo


## 000 STL Random Routines

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


## 018 Binary Tree Traversal Iteration

```c++
#include <iostream>
#include <stack>
#include <vector>

struct Node
{
    Node() = default;
    Node(int v) : val(v) {}
    
    int val {0};
    Node * left {nullptr};
    Node * right {nullptr};
};

// Root Left Right. 
void preOrderTraverse(Node * head)
{
    if (!head) return;

    std::stack<Node *> st;
    st.push(head);

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
void inOrderTraverse(Node * head)
{
    if (!head) return;
    
    std::stack<Node *> st;

    // Stack is empty when resolving root's right subtree. 
    while (!st.empty() || head)
    {
        if (head)
        {
            st.push(head);
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

void postOrderTraverse(Node * head)
{
    if (!head) return;

    std::stack<Node *> st;
    st.push(head);

    // head remains root node until a leaf node gets printed. 
    // After that, head denotes the previous node printed.     
    while (!st.empty())
    {
        Node * cur = st.top();

        if (cur->left && head != cur->left && head != cur->right)
        {
            // Has left subtree and left subtree unresolved.
            st.push(cur->left);
        }
        else if (cur->right && head != cur->right)
        {
            // Has right subtree and right subtree unresolved.
            st.push(cur->right);
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

int main()
{
    std::vector<Node> bb {0, 1, 2, 3, 4, 5, 6, 7};

    Node * head = &bb[1];
    head->left = &bb[2];
    head->right = &bb[3];
    head->left->left = &bb[4];
    head->left->right = &bb[5];
    head->right->left = &bb[6];
    head->right->right = &bb[7];

    preOrderTraverse(head);
    inOrderTraverse(head);
    postOrderTraverse(head);
    
    return EXIT_SUCCESS;
}
```


## 021 Merge Sort

```c++
void merge(int * arr, int lo, int mi, int hi) 
{
    int * a = arr + lo;

    int * b = new int [mi - lo];
    int lb = mi - lo;
    for (int i = 0; i != mi - lo; ++i) b[i] = a[i];

    int * c = arr + mi;
    int lc = hi - mi;

    for (int i = 0, j = 0, k = 0; j < lb || k < lc; )
    {
        if (j < lb && (lc <= k || b[j] <= c[k])) a[i++] = b[j++];
        if (k < lc && (lb <= j || c[k] <  b[j])) a[i++] = c[k++];
    }

    delete [] b;
}

void mergeSort(int * arr, int lo, int hi)
{
    if (hi < lo + 2) return;

    int mi = lo + ((hi - lo) >> 1);
    mergeSort(arr, lo, mi);
    mergeSort(arr, mi, hi);

    merge(arr, lo, mi, hi);
}

void mergeSortIterative(int * arr, int lo, int hi)
{
    if (hi < lo + 2) return;
    
    arr += lo;
    int n = hi - lo;

    // Invoke merge routine sequentially along arr
    // with granularity 1, 2, 4, 8, ...
    for (int step = 1, ll = 0, mi, rr; step < n; step <<= 1, ll = 0)
    {
        while (ll < n)
        {
            mi = ll + step;
            if (n - 1 < mi) break;  // Left part is sorted already. 
            rr = std::min(mi + step, n);
            merge(arr, ll, mi, rr);
            ll = rr;
        }
    }
}
```

## 022 Merge

1. 思考一个问题在大范围上的答案，是否等于，左部分的答案 + 右部分的答案 + 跨越左右产生的答案
2. 计算“跨越左右产生的答案”时，如果加上左、右各自有序这个设定，会不会获得计算的便利性
3. 如果以上两点都成立，那么该问题很可能被归并分治解决（话不说满，因为总有很毒的出题人）
4. 求解答案的过程中只需要加入归并排序的过程即可，因为要让左、右各自有序，来获得计算的便利性

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
// https://www.nowcoder.com/practice/edfe05a1d45c4ea89101d936cac32469
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

// LC 493. Reverse Pairs 
// https://leetcode.cn/problems/reverse-pairs/
int count(int * arr, int lo, int hi) 
{
    if (hi < lo + 2) return 0;
    int mi = (lo + hi) / 2;
    return count(arr, lo, mi) + count(arr, mi, hi) + mergeCount(arr, lo, mi, hi);
}

int mergeCount(int * arr, int lo, int mi, int hi) 
{
    // 统计部分
    int ans = 0;
    
    for (int i = lo, j = mi; i < mi; ++i) 
    {
        while (j < hi && 2 * static_cast<long long>(arr[j]) < static_cast<long long>(arr[i])) ++j;
        ans += j - mi;
    }

    // 正常merge
    merge(arr, lo, mi, hi);

    return ans;
}
```

## 023 Quick Sort

Dutch Flag style quick sort:
```c++
// a[lo, hi], NOTE it's a RIGHT-CLOSE interval!
std::pair<int, int> partition(int * a, int lo, int hi)
{
    int p = a[lo + std::rand() % (hi - lo + 1)];
    int mi = lo;

    while (mi <= hi)
    {
        if (a[mi] < p)       std::swap(a[lo++], a[mi++]);
        else if (a[mi] == p) ++mi;
        else                 std::swap(a[hi--], a[mi]);
    }

    return {lo, hi};
}

void quickSort(int * a, int lo, int hi)
{
    if (hi < lo + 2) return;

    auto [l, r] = partition(a, lo, hi - 1);
    quickSort(a, lo, l);
    quickSort(a, r + 1, hi);
}
```
Legacy quick sort:
```c++
// arr[lo, hi], NOTE that it's a CLOSED inverval! 
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

void quickSort(int * a, int lo, int hi)
{
    if (hi < lo + 2) return;

    int mi = partition(a, lo, hi - 1);
    quickSort(a, lo, mi);
    quickSort(a, mi + 1, hi);
}

void quickSortIterative(int * a, int lo, int hi)
{
    std::stack<std::pair<int, int>> st;
    st.emplace(lo, hi);

    while (!st.empty())
    {
        auto [ll, rr] = st.top();
        st.pop();
        if (rr < ll + 2) continue;

        int mi = partition(a, ll, rr - 1);
        st.emplace(mi + 1, rr);
        st.emplace(ll, mi);
    }
}
```

## 024 Quick Select

```c++
// LC 215. Kth Largest Element in an Array
// https://leetcode.com/problems/kth-largest-element-in-an-array/
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
    while (a[(i - 1) / 2], a[i]) 
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