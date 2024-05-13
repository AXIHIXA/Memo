class Solution
{
public:
    double champagneTower(int poured, int query_row, int query_glass)
    {
        // dp[i][j]: Flow over glass in row i at col j. 
        std::vector dp(101, std::vector<double>(101, 0.0));

        dp[0][0] = poured;

        for (int i = 0; i <= query_row; ++i)
        {
            for (int j = 0; j <= i; ++j)
            {
                if (dp[i][j] < 1.0)
                {
                    continue;
                }

                dp[i + 1][j] += 0.5 * (dp[i][j] - 1.0);
                dp[i + 1][j + 1] += 0.5 * (dp[i][j] - 1.0);
            }
        }

        return std::min(dp[query_row][query_glass], 1.0);
    }
};