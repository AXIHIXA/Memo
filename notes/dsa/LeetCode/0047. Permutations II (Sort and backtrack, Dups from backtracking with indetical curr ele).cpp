class Solution
{
public:
    std::vector<std::vector<int>> permuteUnique(std::vector<int> & nums)
    {
        auto n = static_cast<const int>(nums.size());
        std::sort(nums.begin(), nums.end());

        std::vector<std::vector<int>> ans;
        std::vector<int> tmp;
        std::vector<unsigned char> taken(n, false);
        
        // Dups from backtracking with identical curr element. 
        std::function<void ()> backtrack = [&nums, n, &ans, &tmp, &taken, &backtrack]()
        {
            if (tmp.size() == n)
            {
                ans.push_back(tmp);
                return;
            }

            for (int i = 0, prev = -1; i < n; ++i)
            {
                if (taken[i] || 0 <= prev && nums[i] == nums[prev])
                {
                    continue;
                }

                taken[i] = true;
                tmp.emplace_back(nums[i]);
                prev = i;
                backtrack();
                tmp.pop_back();
                taken[i] = false;
            }
        };

        backtrack();

        return ans;
    }
};