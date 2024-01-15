class Solution 
{
public:
    int firstMissingPositive(vector<int> & nums) 
    {
        std::ios_base::sync_with_stdio(false);
        std::cin.tie(nullptr);
        
        // The smallest missing positive integer must be in range [1, n + 1]. 
        int n = nums.size();

        // Make sure `1` is present in nums.
        bool oneIsPresent = false;

        for (int num : nums)
        {
            if (num == 1)
            {
                oneIsPresent = true;
                break;
            }
        }

        if (!oneIsPresent) return 1;

        // Filter negatives, zeros, or > n. 
        // Now all elements in `nums` are in range [1, n]. 
        for (int & num : nums)
        {
            if (num <= 0 || n < num) num = 1;
        }

        // Use index as a hash key and number sign as a presence detector.
        // For example, if nums[1] is negative,
        // that means that `1` is present in nums. 
        // If nums[2] is positive, then `2` is missing.
        for (int num : nums)
        {
            int a = std::abs(num);
            if (a == n) a = 0;
            nums[a] = -std::abs(nums[a]);
        }

        // Now the index of the first positive number 
        // is equal to first missing positive.
        for (int i = 1; i != n; ++i)
        {
            if (0 < nums[i]) return i;
        }

        if (0 < nums[0]) return n;

        return n + 1;
    }
};