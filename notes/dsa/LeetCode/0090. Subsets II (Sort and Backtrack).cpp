class Solution
{
public:
    std::vector<std::vector<int>> subsetsWithDup(std::vector<int> & nums)
    {
        auto n = static_cast<const int>(nums.size());
        std::sort(nums.begin(), nums.end());

        std::vector<std::vector<int>> ans;
        std::vector<int> tmp;

        std::function<void (int)> backtrack = [&backtrack, &nums, n, &ans, &tmp](int i)
        {
            if (n <= i)
            {
                ans.push_back(tmp);
                return;
            }

            tmp.emplace_back(nums[i]);
            backtrack(i + 1);
            tmp.pop_back();
            
            while (i + 1 < n && nums[i] == nums[i + 1]) ++i;
            backtrack(i + 1);
        };

        backtrack(0);

        return ans;
    }
};