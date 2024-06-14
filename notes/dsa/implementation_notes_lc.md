# Implementation Notes - Coding/Algorithm



## 折腾中间结果：补集思想，拼接思想

- 环形 转换为 反向补集
  - [918. Maximum Sum Circular Subarray](https://leetcode.com/problems/maximum-sum-circular-subarray/)
    - 1D Prefix Sum for intermediate result;
    - Max Sum Circular Subarr EQUALS Max Regular Subarr OR Sum All Minus Min Regular Subarr. 
- 至少 k 的有多少 转换为 反向补集
  - [930. Binary Subarrays With Sum](https://leetcode.com/problems/binary-subarrays-with-sum/)
    - (a) 前缀和 加 哈希表
    - (b) 滑动窗口统计 和最大为 k 以及 k + 1 的子数组 的数量，答案是这俩相减
    - Note that, (b) holds iff. all elements are non-negative!!!
- 找两个 转换为 找所有单个然后拼
  - 发电厂问题：
    - 网格里有被占用的地方，发电厂为正方形
    - 建两个一样大小的发电厂，则这两个发电厂最大能有多大？
    - 2D 前缀 DP 计算每个格子做右下角时建一个发电厂最大能多大，之后暴力拼成两个



## DP

- Single-input Interval DP: `dp[i][j]` denotes answer for `input[i..j]`. 
  - Transfer patterns: 
    - Shrink/extend either/both side(s)
      - Enumerate length and center
        - [5. Longest Palindromic Substring](https://leetcode.com/problems/longest-palindromic-substring/)
      - Sliding window manner
        - [516. Longest Palindromic Subsequence](https://leetcode.com/problems/longest-palindromic-subsequence/)
    - Enumerate length and left side, split in middle
      - Middle split pointss could be alone or part of left interval: 
        - `[i, k) + k + (k, j]`: `i <= k <= j`. 
        - `[i, k] + [k + 1, j]`: `i <= k < j`. 
        - Note **not** all splits are naturally valid for state transfer. 
  - Examples
    - [312. Burst Balloons](https://leetcode.com/problems/burst-balloons/) (HARD)
      - One-padding of input, Enumerate interval length and left node, split in middle
    - [516. Longest Palindromic Subsequence](https://leetcode.com/problems/longest-palindromic-subsequence/) (MEDIUM)
      - Shrink either side
    - [1000. Minimum Cost to Merge Stones](https://leetcode.com/problems/minimum-cost-to-merge-stones/) (HARD)
      - Special care for initialization and step size for split point (`k`). 
      - **Not** all splits are naturally valid for state transfer. 
- Double-input Interval DP: 
  - `dp[i][j][len]` denotes answer for `s1[i:i+len]` and `s2[j:j+len]`. 
    - [87. Scramble String](https://leetcode.com/problems/scramble-string/)
      - Transfer: Split interval in middle...
  - `dp[i][j][...]` denotes answer for `s1[i]` and `s2[j]` with addition status info. 
    - [1397. Find All Good Strings](https://leetcode.com/problems/find-all-good-strings/)
- Bitmask DP with plan info tracking.
  - [943. Find the Shortest Superstring](https://leetcode.com/problems/find-the-shortest-superstring/) (HARD) 



## Subarray

- **Subarray**: Contiguous chunk
  - **Prefix Sum?**
  - **Sliding Window?**
    - "Max K" variant?
- [930. Binary Subarrays With Sum](https://leetcode.com/problems/binary-subarrays-with-sum/)
  - (a) Prefix sum and HashMap. 
  - (b) Sliding window of number of subarrays summing up to at most goal, prefix sum manner minus. 
    - Note that, (b) holds iff. all elements are non-negative!!!
- [560. Subarray Sum Equals K](https://leetcode.com/problems/subarray-sum-equals-k/)
  - Note that this could be done only with prefix sum + HashMap method. 
  - Prefix sum could be computed with rolling number manner in O(1) space. 
  - Note that (b) for 930 does **not** work for this problem!!!
- [992. Subarrays with K Different Integers](https://leetcode.com/problems/subarrays-with-k-different-integers/)
  - Still non-negative counter constitutions, sliding window at most k MINUS at most k - 1. 
- [2602. Minimum Operations to Make All Array Elements Equal](https://leetcode.com/problems/minimum-operations-to-make-all-array-elements-equal/)
  - Sort and Prefix Sum. 



## Random Element Removal for A Heap-like Interface

- Heap and HashSet: Lazy Removal / Delayed Removal
  - HashSet records deleted elements in heap;
  - Each time getting elements from heap, pop until top is not marked deleted. 
- Or use a **MULTISET** RBTree...
  - Note it MUST be `std::multiset`, which allows duplicates!
  - Also note that `std::multiset::erase(key)` removes **ALL** elements with key `key`!
    - `find` returns an iterator to a random element with key `key`, `erase(find(key))` should work. 
    - `std::multiset` also provides `lower_bound`, `upper_bound`, `equal_range`. 
- [218. The Skyline Problem](https://leetcode.com/problems/the-skyline-problem/)
  - Vertical Scanline with lazy removal optimization. 
- [716. Max Stack](https://leetcode.com/problems/max-stack/)



## BFS

- So the BFS queue could contain
  - Either "raw" states (unprocessed), 
  - Or "done"/"processed" states
    - Processed before enqueued, 
    - Usually when we have outer memory tables, e.g:
- [864. Shortest Path to Get All Keys](https://leetcode.com/problems/shortest-path-to-get-all-keys/)
  - State compression BFS, enqueue done/processed states.  
  - Prune with recorded min step vector for each (row, col, state). 
  - **Handle updates in direction loop** s.t. sub-optimal steps won't be added to queue. 

