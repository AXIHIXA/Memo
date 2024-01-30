class Solution
{
public:
    int numberOfWays(int startPos, int endPos, int k)
    {
        if (k == 0) return startPos == endPos;
        
        // dp[i][j] denotes #ways to move to 
        // pos where (|pos - startPos| == j) in exactly i steps. 
        static constexpr int p = 1e9 + 7;
        int dp[1001][1001] {0};

        for (int i = 1; i <= k; ++i)
        {
            dp[i][i] = 1;

            for (int j = 0; j < i; ++j)
            {
                dp[i][j] = (dp[i - 1][std::abs(j - 1)] + dp[i - 1][j + 1]) % p;
            }
        }

        return dp[k][std::abs(endPos - startPos)];
    }
};