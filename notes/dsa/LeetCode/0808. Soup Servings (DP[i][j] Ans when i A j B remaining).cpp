class Solution
{
public:
    double soupServings(int n)
    {
        // An n > 200 * 25 ml could be simulated by n == 200 * 25 at 1e-5 precision. 
        n = std::min(200, static_cast<int>(std::ceil(n / 25.0)));
        
        // 1. 4A 0B
        // 2. 3A 1B
        // 3. 2A 2B
        // 4. 1A 3B

        // dp[i][j]: 
        // Ans probability when we have i units of A and j units of B. 
        std::vector dp(n + 10, std::vector<double>(n + 10, 0.0));
        dp[0][0] = 0.5;

        for (int j = 1; j <= n; ++j)
        {
            dp[0][j] = 1.0;
        }

        for (int i = 1; i <= n; ++i)
        {
            for (int j = 1; j <= n; ++j)
            {
                double a = dp[std::max(i - 4, 0)][j];
                double b = dp[std::max(i - 3, 0)][std::max(j - 1, 0)];
                double c = dp[std::max(i - 2, 0)][std::max(j - 2, 0)];
                double d = dp[std::max(i - 1, 0)][std::max(j - 3, 0)];
                dp[i][j] = 0.25 * (a + b + c + d);
            }
        }

        return dp[n][n];
    }
};