class Solution
{
public:
    int minimizedMaximum(int n, std::vector<int> & quantities)
    {
        auto good = [n, &quantities](int num) -> bool
        {
            static bool allZero = std::all_of(
                    quantities.cbegin(), 
                    quantities.cend(), 
                    [](const int x) { return x == 0; }
            );
            
            if (num == 0)
            {
                return allZero;
            }

            int storesNeeded = 0;

            for (int x : quantities)
            {
                storesNeeded += (x + num - 1) / num;

                if (n < storesNeeded)
                {
                    return false;
                }
            }

            return true;
        };

        int lo = 0;
        int hi = *std::max_element(quantities.cbegin(), quantities.cend()) + 1;
        int ans = hi;

        while (lo < hi)
        {
            int mi = lo + ((hi - lo) >> 1);

            if (good(mi))
            {
                ans = mi;
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