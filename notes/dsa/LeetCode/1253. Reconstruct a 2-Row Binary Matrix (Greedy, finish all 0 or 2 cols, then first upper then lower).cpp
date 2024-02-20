class Solution
{
public:
    std::vector<std::vector<int>> reconstructMatrix(int upper, int lower, std::vector<int> & colsum)
    {
        auto n = static_cast<const int>(colsum.size());

        if (upper + lower != std::accumulate(colsum.cbegin(), colsum.cend(), 0))
        {
            return {};
        }

        std::vector ans(2, std::vector<int>(n, 0));

        for (int i = 0; i < n; ++i)
        {
            if (colsum[i] == 2)
            {
                if (--upper < 0 || --lower < 0)
                {
                    return {};
                }
                
                ans[0][i] = ans[1][i] = 1;
            }
        }

        for (int i = 0; i < n; ++i)
        {
            if (colsum[i] != 1)
            {
                continue;
            }
            
            // Greedy construction, upper row first then lower. 
            if (0 < upper) 
            {
                ans[0][i] = 1;
                --upper;
            }
            else if (0 < lower)
            {
                ans[1][i] = 1;
                --lower;
            }
            else
            {
                return {};
            }
        }

        return ans;
    }
};