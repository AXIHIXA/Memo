# Implementation Notes - STL



## OJ Routines

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

int getInt()
{
    int x = 0, c = std::getchar();

    while (!isdigit(c))
    {
        c = std::getchar();
    }

    while (isdigit(c))
    {
        x = x * 10 + (c ^ 48);
        c = std::getchar();
    }

    return x;
}
```



## STL Random

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


## Bin Search

- **Partitioned**:
  - Former half evaluates to `false`, latter half evaluates to `true` w.r.t. `comp(target, e)` or `t < e`. 
- For a sorted range `a` (non-decreasing, sorted with `<`):
  - `std::upper_bound(a.crbegin(), a.crend(), t)`: Leftmost element s.t. `t < e`; 
  - `std::lower_bound(a.crbegin(), a.crend(), t)`: Leftmost element s.t. `t <= e`; 
  - `std::upper_bound(a.crbegin(), a.crend(), t, std::greater<>())`: Rightmost element s.t. `e < t`;
  - `std::lower_bound(a.crbegin(), a.crend(), t, std::greater<>())`: Rightmost element s.t. `e <= t`;



## Prefix Sum

- [std::partial_sum](https://en.cppreference.com/w/cpp/algorithm/partial_sum) overflows!
  - A variable of type `InputIt`'s value type, is used as accumulator for intermediate results.
  - Providing `BinaryOperation` predicates like `std::plus<long long>` for `std::vector<int>`s would **not** help.
- [std::inclusive_scan](https://en.cppreference.com/w/cpp/algorithm/inclusive_scan) avoids this problem by providing `init` accumulator.
- [std::transform_inclusive_scan](https://en.cppreference.com/w/cpp/algorithm/transform_inclusive_scan) is also an option. 
```c++
// A large vector, prefix sum overflows int32. 
std::vector<int> nums = { /* ... */ };
auto n = static_cast<const int>(nums.size());
std::vector<long long> ps1(n + 1, 0), ps2(n + 1, 0), ps3(n + 1, 0), ps4(n + 1, 0);

// Good. 
for (int i = 0; i < n; ++i) ps1[i + 1] = ps1[i] + nums[i];                              

// OVERFLOWS!!!
std::partial_sum(nums.cbegin(), nums.cend(), ps2.begin() + 1, std::plus<long long>());  

// Also good. (since C++17)
std::inclusive_scan(
        nums.cbegin(), nums.cend(), ps3.begin() + 1, 
        std::plus<>(), 0LL
);

// Also good. (since C++17)
std::transform_inclusive_scan(
        nums.cbegin(), nums.cend(), ps4.begin() + 1, 
        std::plus<>(), [](int x) { return static_cast<long long>(x); }, 0LL
);                                                                                      
```



## String Concatenation

- While serializing one should avoid concatenating strings in programming languages where strings are immutable 
  - Concatenating strings takes linear time. 
  - If we concatenate a string of length `O(N)` about `O(N)` times, then the total time complexity will be `O(N**2)`.
- Instead, one should use a `StringBuilder` or `StringBuffer` in `Java`, or `list` in `Python`. 
- Now, `std::string` in `C++` is mutable, so we can concatenate strings. 
  - `operator +` **wouldn't** create a new string each time. 
  - It would just append the new string to the existing string. 
  - Therefore, we can use `operator +` to concatenate strings in `C++`.
- [Reference](https://leetcode.com/problems/subtree-of-another-tree/editorial/)



## Static Arrays

- Do **not** allocate large static arrays in class A and subclass A into another class B!
- This leads to the [relocation truncated to fit](https://www.technovelty.org/c/relocation-truncated-to-fit-wtf.html) linker error.
- Sample code. Use `std::vector` for this `Trie`'s storage!: 
```c++
class Trie
{
public:
    friend class Solution;
    
public:
    ...

private:
    static constexpr int kMaxSize = 128'000'000;

    static int cnt;
    static std::array<std::array<int, 26>, kMaxSize> tree;
    static std::array<int, kMaxSize> endd;
};

static int Trie::cnt = 1;
static std::array<std::array<int, 26>, Trie::kMaxSize> tree = {0};
static std::array<int, Trie::kMaxSize> endd = {0};

class Solution
{
public:
    ...

private:
    static Trie trie;
};

Trie Solution::trie;
```