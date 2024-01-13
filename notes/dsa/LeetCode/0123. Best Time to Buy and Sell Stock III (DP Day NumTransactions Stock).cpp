class Solution 
{
public:
    int maxProfit(vector<int> & prices) 
    {
        auto n = static_cast<int>(prices.size());

        std::vector<int> dpData(n * 3 * 2, kNegativeInt);

        auto dp = [n, &dpData](int d, int t, int s) -> int &
        {
            return dpData[d * 6 + t * 2 + s];
        };

        dp(0, 0, 0) = 0;
        dp(0, 1, 1) = -prices[0];

        for (int d = 1; d != n; ++d)
        {
            for (int t = 0; t < 3; ++t)
            {
                dp(d, t, 0) = max(dp(d - 1, t, 0), dp(d - 1, t, 1) + prices[d]);

                if (0 < t)
                {
                    dp(d, t, 1) = max(dp(d - 1, t, 1), dp(d - 1, t - 1, 0) - prices[d]);
                }
            }
        }

        int res = 0;
        
        for (int t = 0; t < 3; ++t)
        {
            res = max(res, dp(n - 1, t, 0));
        }

        return res;
    }

private:
    static constexpr int kNegativeInt = -1000000000;
};