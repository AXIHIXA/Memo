class Solution
{
public:
    int minEatingSpeed(std::vector<int> & piles, int h)
    {
        auto n = static_cast<const int>(piles.size());
        
        // argmin k s.t. sum( [ (p[i] + k - 1) / k for i in [0...n) ] ) == h. 
        auto get = [n, &piles](int k)
        {
            int hours = 0;
            
            for (int i = 0; i < n; ++i)
            {
                hours += (piles[i] + k - 1) / k;
            }

            return hours;
        };

        int lo = 1;
        int hi = 1 + *std::max_element(piles.cbegin(), piles.cend());

        while (lo < hi)
        {
            int mi = lo + ((hi - lo) >> 1);
            int time = get(mi);

            if (h < time)
            {
                lo = mi + 1;
            }
            else
            {
                hi = mi;
            }
        }

        return lo;
    }
};

// e, t 1st e[i] s.t. e[i] == h
// + + + + = = = = - - - -
// 