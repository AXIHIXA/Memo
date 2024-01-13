class Solution 
{
public:
    int maxProfit(vector<int> & prices) 
    {
        auto n = static_cast<int>(prices.size());

        dp[0][0] = 0;
        dp[0][1] = 0;
        dp[0][2] = -prices[0];

        for (int d = 1; d != n; ++d)
        {
            dp[d][0] = dp[d - 1][2] + prices[d];
            dp[d][1] = max(dp[d - 1][0], dp[d - 1][1]);
            dp[d][2] = max(dp[d - 1][2], dp[d - 1][1] - prices[d]);
        }

        return max(dp[n - 1][0], dp[n - 1][1]);
    }

private:
    static constexpr int kMaxPricesLen = 5000;
    
    // dp[d][s]: max profit by end of day
    // s == 0: not holding the stock (sold today);
    // s == 1: not holding the stock (sold earlier);
    // s == 2: holding the stock. 
    // Viable transformations:
    // [d - 1][0] -> [d][1]
    // [d - 1][1] -> [d][1], [d][2] (- prices[d])
    // [d - 1][2] -> [d][2], [d][0] (+ prices[d])
    int dp[kMaxPricesLen + 10][3] {0};
};