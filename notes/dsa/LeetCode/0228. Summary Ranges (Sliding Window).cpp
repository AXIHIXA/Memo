class Solution
{
public:
    std::vector<std::string> summaryRanges(std::vector<int> & nums)
    {
        std::vector<std::string> ans;

        for (int i = 0; i != nums.size(); ++i)
        {
            int start = nums[i];
            while (i + 1 < nums.size() && nums[i] + 1 == nums[i + 1]) ++i;
            ans.emplace_back(std::to_string(start));
            if (start != nums[i]) ans.back() += "->" + std::to_string(nums[i]);
        }

        return ans;
    }
};