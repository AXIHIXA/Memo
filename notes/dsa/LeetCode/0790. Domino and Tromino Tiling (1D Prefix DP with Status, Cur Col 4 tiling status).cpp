class Solution
{
public:
    int numTilings(int n)
    {
        // dp[i][j]: 
        // Number of ways to tile i columns (1-indexed)
        // with columns [0...i - 1] tiled already. 
        // j is status of i-1-th and i-th column: 
        // j == 0  j == 1  j == 2  j == 3
        // X       XX      XX      X 
        // X       XX      X       XX
        std::vector dp(n + 10, std::vector<int>(4, 0));
        dp[1][0] = 1;
        dp[1][1] = 1;

        for (int i = 2; i <= n; ++i)
        {
            dp[i][0] = dp[i - 1][1];
            
            int cur = 0;

            for (int j = 0; j < 4; ++j)
            {
                cur = (cur + dp[i - 1][j]) % p;
            }

            dp[i][1] = cur;
            dp[i][2] = (dp[i - 1][0] + dp[i - 1][3]) % p;
            dp[i][3] = (dp[i - 1][0] + dp[i - 1][2]) % p;
        }

        return dp[n][1];
    }

private:
    static constexpr int p = 1'000'000'007;
};