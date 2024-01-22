class Solution
{
public:
    bool containsNearbyDuplicate(vector<int> & nums, int k)
    {
        std::unordered_set<int> hs;

        for (int i = 0; i != nums.size(); ++i)
        {
            auto it = hs.find(nums[i]);
            if (it != hs.end()) return true;
            hs.emplace_hint(it, nums[i]);

            if (k < hs.size()) hs.erase(nums[i - k]);
        }

        return false;
    }
};