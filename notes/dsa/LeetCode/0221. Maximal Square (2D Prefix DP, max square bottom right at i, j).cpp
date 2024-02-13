class Solution
{
public:
    int maximalSquare(std::vector<std::vector<char>> & matrix)
    {
        auto m = static_cast<int>(matrix.size());
        auto n = static_cast<int>(matrix.front().size());

        // dp[i][j]: Max square length with bottom right element at (i, j). 
        std::vector<std::vector<int>> dp(m, std::vector<int>(n, 0));
        
        int maxLen = 0;
        
        for (int i = 0; i < m; ++i) if (matrix[i][0] - '0') dp[i][0] = 1, maxLen = 1;
        for (int j = 0; j < n; ++j) if (matrix[0][j] - '0') dp[0][j] = 1, maxLen = 1;

        for (int i = 1; i < m; ++i)
        {
            for (int j = 1; j < n; ++j)
            {
                if (matrix[i][j] - '0')
                {
                    dp[i][j] = 1 + std::min({dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1]});
                    maxLen = std::max(maxLen, dp[i][j]);
                }
            }
        }

        return maxLen * maxLen;
    }
};