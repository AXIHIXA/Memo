class Solution
{
public:
    int minimumSize(std::vector<int> & nums, int maxOperations)
    {
        auto good = [&nums](int p, int maxOp) -> bool
        {
            int op = 0;

            for (int x : nums)
            {
                op += (x + p - 1) / p - 1;
                
                if (maxOp < op)
                {
                    return false;
                }
            }

            return true;
        };

        int lo = 1;
        int hi = *std::max_element(nums.cbegin(), nums.cend()) + 1;
        int ans = hi;

        while (lo < hi)
        {
            int mi = lo + ((hi - lo) >> 1);

            if (good(mi, maxOperations))
            {
                ans = std::min(ans, mi);
                hi = mi;
            }
            else
            {
                lo = mi + 1;
            }
        }
        
        return ans;
    }
};