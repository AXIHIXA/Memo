class Solution
{
public:
    std::vector<int> nextGreaterElement(std::vector<int> & nums1, std::vector<int> & nums2)
    {
        auto m = static_cast<const int>(nums1.size());
        auto n = static_cast<const int>(nums2.size());

        // Monotonic Stack (strictly decreasing) AND HashMap. 
        // Again, If Nums2: ..., [4], 3, 2, 1, 0, ... 
        // 4 is gonna be pushed into stack (top == 3), 
        // 3 is not gonna be answer for any x to the left part of nums2. 
        std::unordered_map<int, int> hashMap;
        std::stack<int> stk;

        for (int i = n - 1; 0 <= i; --i)
        {
            while (!stk.empty() && stk.top() <= nums2[i])
            {
                stk.pop();
            }

            hashMap.emplace(nums2[i], stk.empty() ? -1 : stk.top());
            stk.emplace(nums2[i]);
        }

        std::vector<int> ans(m);

        for (int i = 0; i < m; ++i)
        {
            ans[i] = hashMap.at(nums1[i]);
        }

        return ans;
    }
};