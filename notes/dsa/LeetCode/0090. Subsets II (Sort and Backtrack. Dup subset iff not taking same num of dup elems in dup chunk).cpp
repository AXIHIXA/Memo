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

            // Take this element. 
            tmp.emplace_back(nums[i]);
            backtrack(i + 1);
            tmp.pop_back();

            // Does not take this element (if this ele is not a dup); 
            // or ignoring 1, 2, ..., numDup copies of this element. 
            // Note that dup subsets come from
            // not taking same number of dup elements in the duo chunk. 
            // E.g., 1 2 2 3 ..., 
            // Not taking 1st 2 ans taking all other 2s, 
            // Not taking 2nd 2 and taking all other 2s results in a dup subset. 
            while (i + 1 < n && nums[i] == nums[i + 1]) ++i;
            backtrack(i + 1);
        };

        backtrack(0);

        return ans;
    }
};