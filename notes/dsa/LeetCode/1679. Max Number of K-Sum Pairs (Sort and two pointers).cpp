class Solution
{
public:
    int maxOperations(std::vector<int> & nums, int k)
    {
        auto n = static_cast<int>(nums.size());
        if (n == 1) return 0;
        if (n == 2) return nums[0] + nums[1] == k;

        std::sort(nums.begin(), nums.end());

        int ans = 0;

        for (int ll = 0, rr = n - 1; ll < rr; )
        {
            if (nums[ll] + nums[rr] < k) ++ll;
            else if (k < nums[ll] + nums[rr]) --rr;
            else { ++ll, --rr, ++ans; }
        }

        return ans;
    }
};