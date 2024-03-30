class Solution
{
public:
    long long maxRunTime(int n, std::vector<int> & batteries)
    {
        long long totalCapacity = std::reduce(batteries.cbegin(), batteries.cend(), 0LL);
        long long maxCapacity = *std::max_element(batteries.cbegin(), batteries.cend());

        if (maxCapacity * n < totalCapacity)
        {
            return totalCapacity / n;
        }

        auto good = [n, &batteries](long long t) -> bool 
        {
            long long fragmentCapacity = 0LL;
            long long unfedComputers = n;

            for (int x : batteries)
            {
                if (t < x)
                {
                    --unfedComputers;
                }
                else
                {
                    fragmentCapacity += x;
                }

                if (unfedComputers * t <= fragmentCapacity)
                {
                    return true;
                }
            }

            return false;
        };

        long long ans = 0LL;

        for (long long lo = 0LL, hi = totalCapacity / n + 1LL; lo < hi; )
        {
            long long mi = lo + ((hi - lo) >> 1LL);

            if (good(mi))
            {
                ans = mi;
                lo = mi + 1LL;
            }
            else
            {
                hi = mi;
            }
        }

        return ans;
    }
};