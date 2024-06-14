class Solution
{
public:
    int maxProduct(std::vector<int> & nums)
    {
        auto n = static_cast<const int>(nums.size());
        double curMax = nums[0];
        double curMin = nums[0];
        double ans = nums[0];

        for (int i = 1; i < n; ++i)
        {
            double cur = nums[i];
            double tmpMax = std::max(cur, std::max(curMax * cur, curMin * cur));
            double tmpMin = std::min(cur, std::min(curMax * cur, curMin * cur));
            curMax = tmpMax;
            curMin = tmpMin;
            ans = std::max(ans, curMax);
        }

        return static_cast<int>(ans);
    }
};