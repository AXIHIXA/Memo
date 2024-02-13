class Solution
{
public:
    int maxProfit(std::vector<int> & prices)
    {
        auto n = static_cast<int>(prices.size());
        if (n == 1) return 0;

        // Cost of this transaction, profit of this transaction. 
        std::pair<int, int> t1 {kLargeInt, 0};
        std::pair<int, int> t2 {kLargeInt, 0};

        for (int p : prices)
        {
            // The maximum profit if only one transaction is allowed. 
            t1.first = std::min(t1.first, p);
            t1.second = std::max(t1.second, p - t1.first);

            // Re-invest the gained profit in the second transaction. 
            t2.first = std::min(t2.first, p - t1.second);
            t2.second = std::max(t2.second, p - t2.first);
        }
        
        return t2.second; 
    }

private:
    static constexpr int kLargeInt =  1'000'000'000;
    static constexpr int kSmallInt = -1'000'000'000;

    static int maxProfitDp(std::vector<int> & prices)
    {
        auto n = static_cast<int>(prices.size());
        if (n == 1) return 0;

        // dp[d][t][s]: Max profit at end of day i, 0 <= i < n. 
        // t: Number of transactions (buy-sell-buy-sell) made, 0 <= t <= 4. 
        std::vector dp(n, std::vector(5, kSmallInt));
        dp[0][0] = 0;
        dp[0][1] = -prices[0];

        int ans = 0;

        for (int d = 1; d < n; ++d)
        {
            dp[d][0] = 0;
            dp[d][1] = std::max(dp[d - 1][1], dp[d - 1][0] - prices[d]);
            dp[d][2] = std::max(dp[d - 1][2], dp[d - 1][1] + prices[d]);
            dp[d][3] = std::max(dp[d - 1][3], dp[d - 1][2] - prices[d]);
            dp[d][4] = std::max(dp[d - 1][4], dp[d - 1][3] + prices[d]);
            ans = std::max(ans, *std::max_element(dp[d].cbegin(), dp[d].cend()));
        }

        return ans; 
    }
};