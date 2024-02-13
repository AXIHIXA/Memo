class Solution 
{
public:
    int maxProfit(int k, std::vector<int> & prices)
    {
        auto n = static_cast<int>(prices.size());
        if (n == 1) return 0;

        // dp[d][t]: Max profit by end of day d. 
        // t: Number of transactions (buy-sell-buy-sell-buy-...) made. 
        // (t & 1) == 1 means holding stock, == 0 means not holding stock. 
        std::vector dp(n, std::vector(1 + k * 2, kSmallInt));
        dp[0][0] = 0;
        dp[0][1] = -prices[0];

        int ans = 0;

        for (int d = 1; d < n; ++d)
        {
            dp[d][0] = 0;

            for (int t = 1; t <= k * 2; ++t)
            {
                dp[d][t] = std::max(
                        dp[d - 1][t], 
                        dp[d - 1][t - 1] + ((t & 1) ? -prices[d] : prices[d])
                );
            }

            ans = std::max(ans, *std::max_element(dp[d].cbegin(), dp[d].cend()));
        }

        return ans;
    }

private:
    static constexpr int kSmallInt = -1'000'000'000;

    // Let dp(d, t, s) denote the max profit by end of day d, 
    // with t [purchases] made, 
    // and currently holding (s == 1) or not holding (s == 0) the stock. 
    // At day d, we have four possible actions:
    // (1) If we are holding stock by end of day d:
    //     (1.1) Keep holding the stock, 
    //           dp(d, t, 1) = dp(d - 1, t, 1);
    //     (1.2) Sell current stock, 
    //           dp(d, t, 0) = dp(d - 1, t, 1) + prices[d];
    // (2) If we are not holding the stock by end of day d:
    //     (2.1) Keep not holding the stock, 
    //           dp(d, t, 0) = dp(d - 1, t, 0);
    //     (2.2) Buy in the stock (possible iff. 0 < t), 
    //           dp(d, t, 1) = dp(d - 1, t - 1, 0) - prices[d];
    // We use t as #purchases because the restraint is on purchases, 
    // e.g., could purchase only after a sale. 
    int maxProfitDp2(int k, std::vector<int> & prices) 
    {
        auto n = static_cast<int>(prices.size());

        if (n <= 0 or k <= 0)
        {
            return 0;
        }

        if (n < k / 2)
        {
            int res = 0;

            for (int d = 1; d != n; ++d)
            {
                res += std::max(0, prices[d] - prices[d - 1]);
            }

            return res;
        }

        std::vector<int> data(n * (k + 1) * 2, kSmallInt);

        auto dp = [n, k, &data](int d, int t, int s) -> int &
        {
            return data[d * ((k + 1) * 2) + t * 2 + s];
        };
            
        dp(0, 0, 0) = 0;
        dp(0, 1, 1) = -prices[0];

        for (int d = 1; d != n; ++d)
        {
            for (int t = 0; t <= k; ++t)
            {
                if (0 < t)
                {
                    dp(d, t, 1) = std::max(dp(d - 1, t, 1), dp(d - 1, t - 1, 0) - prices[d]);
                }

                dp(d, t, 0) = std::max(dp(d - 1, t, 0), dp(d - 1, t, 1) + prices[d]);
            }
        }

        int res = 0;

        for (int t = 0; t <= k; ++t)
        {
            res = std::max(res, dp(n - 1, t, 0));
        }

        return res;
    }
};
