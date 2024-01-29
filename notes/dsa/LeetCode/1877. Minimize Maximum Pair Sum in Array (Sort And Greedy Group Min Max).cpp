class Solution 
{
public:
    int minPairSum(std::vector<int> & nums) 
    {
        // Two pointer greedy. 
        // Suppose we have a1 <= ai <= aj <= an in non-decreasing order. 
        // Contradiction: 
        // Suppose pair [(a1, ai), (aj, an)] is more optimal than 
        // the Greedy result [(a1, an), (ai, aj)]. 
        // However this is impossible, because
        // max [(a1, ai), (aj, an)] is aj + an, 
        // which is no smaller than Greedy result, max [(a1, ai), (aj, an)]. 
        std::sort(nums.begin(), nums.end());

        int ans = nums.front() + nums.back();

        for (int i = 1; i < (nums.size() >> 1); ++i)
        {
            ans = max(ans, nums[i] + nums[nums.size() - 1 - i]);
        }

        return ans;
    }
};