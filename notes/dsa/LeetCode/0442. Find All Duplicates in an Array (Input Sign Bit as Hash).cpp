class Solution
{
public:
    std::vector<int> findDuplicates(std::vector<int> & nums)
    {
        std::vector<int> ans;

        for (int xx : nums)
        {
            int x = std::abs(xx);
            
            if (nums[x - 1] < 0)
            {
                ans.emplace_back(x);
            }

            nums[x - 1] *= -1;
        }

        return ans;
    }
};