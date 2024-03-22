class Solution
{
public:
    std::vector<int> sortArrayByParityII(std::vector<int> & nums)
    {
        auto n = static_cast<const int>(nums.size());

        for (int evenIndex = 0, oddIndex = 1; evenIndex < n && oddIndex < n; )
        {
            if (nums.back() & 1)
            {
                std::swap(nums[oddIndex], nums.back());
                oddIndex += 2;
            }
            else
            {
                std::swap(nums[evenIndex], nums.back());
                evenIndex += 2;
            }
        }

        return std::move(nums);
    }
};