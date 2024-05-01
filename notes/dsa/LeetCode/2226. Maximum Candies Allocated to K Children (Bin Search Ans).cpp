class Solution
{
public:
    int maximumCandies(std::vector<int> & candies, long long k)
    {
        auto good = [&candies, k](long long cp) -> bool
        {
            if (cp == 0LL)
            {
                return true;
            }
            
            long long childrenDone = 0LL;

            for (int x : candies)
            {
                childrenDone += x / cp;

                if (k <= childrenDone)
                {
                    return true;
                }
            }

            return k <= childrenDone;
        };

        long long lo = 0LL;
        long long hi = *std::max_element(candies.cbegin(), candies.cend()) + 1LL;
        long long ans = 0LL;

        while (lo < hi)
        {
            long long mi = lo + ((hi - lo) >> 1LL);

            if (good(mi))
            {
                ans = std::max(ans, mi);
                lo = mi + 1LL;
            }
            else
            {
                hi = mi;
            }
        }

        return static_cast<int>(ans);
    }
};