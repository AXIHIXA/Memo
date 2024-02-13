class Solution
{
public:
    bool increasingTriplet(std::vector<int> & nums)
    {
        auto n = static_cast<int>(nums.size());
        if (n < 3) return false;

        int a = nums[0];
        int b = std::numeric_limits<int>::max();

        for (int i = 1; i < n; ++i)
        {
            if (a < nums[i])
            {
                if (b < nums[i]) return true;
                else b = nums[i];
            }
            else
            {
                a = nums[i];
            }
        }

        return false;
    }
};