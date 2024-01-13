class Solution 
{
public:
    int subarraySum(vector<int> & nums, int k) 
    {
        int ans = 0, sum = 0;

        // Cumulative sum cs[k] = sum(nums[0]...nums[k]). 
        // cs[j] - s[i] == k means sum(nums[i + 1]...nums[j]) == k.

        unordered_map<int, int> mp;
        mp.emplace(0, 1);

        for (int n : nums)
        {
            sum += n;

            if (auto it = mp.find(sum - k); it != mp.end())
            {
                ans += it->second;
            }

            if (auto it = mp.find(sum); it != mp.end())
            {
                ++(it->second);
            }
            else
            {
                mp.emplace(sum, 1);
            }
        }

        return ans;
    }
};