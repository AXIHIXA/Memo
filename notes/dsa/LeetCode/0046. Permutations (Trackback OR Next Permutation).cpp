class Solution 
{
public:
    std::vector<std::vector<int>> permute(std::vector<int> & nums) 
    {
        auto n = static_cast<const int>(nums.size());
        std::vector<std::vector<int>> ans;
        std::vector<int> tmp;
        std::vector<unsigned char> taken(n, false);

        std::function<void ()> backtrack = [&nums, n, &ans, &tmp, &taken, &backtrack]()
        {
            if (tmp.size() == n)
            {
                ans.push_back(tmp);
                return;
            }

            for (int i = 0; i < n; ++i)
            {
                if (taken[i]) continue;

                taken[i] = true;
                tmp.emplace_back(nums[i]);
                backtrack();
                tmp.pop_back();
                taken[i] = false;
            }
        };

        backtrack();

        return ans;
    }
};