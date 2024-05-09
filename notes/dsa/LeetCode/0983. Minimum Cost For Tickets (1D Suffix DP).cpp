class Solution
{
public:
    int mincostTickets(std::vector<int> & days, std::vector<int> & costs)
    {
        auto n = static_cast<const int>(days.size());

        // dp[i]: Min cost to cover days[i:]. 
        std::vector<int> dp(n + 1, std::numeric_limits<int>::max());
        dp[n - 1] = *std::min_element(costs.cbegin(), costs.cend());
        dp[n] = 0;

        for (int i = n - 2; 0 <= i; --i)
        {
            dp[i] = dp[n - 1] + dp[i + 1];
            int j = i + 1;
            while (j < n && days[j] < days[i] + 7) ++j;
            dp[i] = std::min(dp[i], costs[1] + dp[j]);
            while (j < n && days[j] < days[i] + 30) ++j;
            dp[i] = std::min(dp[i], costs[2] + dp[j]);
        }

        return dp[0];
    }
};