class Solution
{
public:
    bool containsNearbyAlmostDuplicate(vector<int> & nums, int indexDiff, int valueDiff)
    {
        std::multiset<long long> window;

        for (int i = 0; i < nums.size(); ++i)
        {
            if (indexDiff < i) window.erase(nums[i - indexDiff - 1]);
            auto it = window.lower_bound(static_cast<long long>(nums[i]) - static_cast<long long>(valueDiff));
            if (it != window.end() && *it - static_cast<long long>(nums[i]) <= static_cast<long long>(valueDiff)) return true;
            window.insert(nums[i]);
        }

        return false;
    }
};