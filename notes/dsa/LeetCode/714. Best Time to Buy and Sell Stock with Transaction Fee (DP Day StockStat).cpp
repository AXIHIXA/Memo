class Solution 
{
public:
    int maxProfit(vector<int> & prices, int fee) 
    {
        auto n = static_cast<int>(prices.size());

        std::vector<int> vec(n * 2, kNegativeInt);

        // dp(d, s) denotes max profit by end of day d.
        // s == 1 denotes we are holding the stock, 0 denotes otherwise. 
        auto dp = [n, &vec](int d, int s) -> int &
        {
            return vec[d * 2 + s];
        };

        dp(0, 0) = 0;
        dp(0, 1) = -prices[0] - fee;

        for (int d = 1; d != n; ++d)
        {
            dp(d, 0) = max(dp(d - 1, 0), dp(d - 1, 1) + prices[d]);
            dp(d, 1) = max(dp(d - 1, 1), dp(d - 1, 0) - prices[d] - fee);

            // cout << "dp(" << d << ", 0) = " 
            //      << "max(" << dp(d - 1, 0) << ", " <<  dp(d - 1, 1) + prices[d] - fee
            //      << " = " << dp(d, 0) << '\n';
            // cout << "dp(" << d << ", 1) = " 
            //      << "max(" << dp(d - 1, 1) << ", " << dp(d - 1, 0) - prices[d] - fee
            //      << " = " << dp(d, 1) << '\n';
        }

        return dp(n - 1, 0);
    }

private:
    static constexpr int kNegativeInt = -1000000000;
};