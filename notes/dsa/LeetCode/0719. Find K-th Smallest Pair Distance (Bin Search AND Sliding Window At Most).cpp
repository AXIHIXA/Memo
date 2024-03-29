class Solution
{
public:
    int smallestDistancePair(std::vector<int> & nums, int k)
    {
        auto n = static_cast<const int>(nums.size());
        std::sort(nums.begin(), nums.end());

        auto numPairsAtMostX = [n, &nums](int x) -> int
        {
            int ans = 0;

            for (int ll = 0, rr = 0; ll < n; ++ll)
            {
                while (rr + 1 < n && nums[rr + 1] - nums[ll] <= x)
                {
                    ++rr;
                }

                ans += rr - ll;
            }

            return ans;
        };
        
        int ans = 0;

        for (int lo = 0, hi = nums[n - 1] - nums[0] + 1; lo < hi; )
        {
            int mi = lo + ((hi - lo) >> 1);
            int rank = numPairsAtMostX(mi);

            if (rank < k)
            {
                lo = mi + 1;
            }
            else
            {
                ans = mi;
                hi = mi;
            }
        }

        return ans;
    }
};