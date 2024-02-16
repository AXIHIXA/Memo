class Solution
{
public:
    std::vector<std::vector<int>> updateMatrix(std::vector<std::vector<int>> & mat)
    {
        auto m = static_cast<const int>(mat.size());
        auto n = static_cast<const int>(mat.front().size());

        // dp[i][j]: Min distance of mat[i][j] to a zero cell. 
        std::vector<std::vector<int>> dp(m, std::vector<int>(n));
        for (int i = 0; i < m; ++i) std::copy(mat[i].begin(), mat[i].end(), dp[i].begin());

        // The first pass: Pretend that we can only move down or right. 
        for (int i = 0; i < m; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                if (!dp[i][j]) continue;

                int minNeighbor = m * n;
                if (0 < i) minNeighbor = std::min(minNeighbor, dp[i - 1][j]);
                if (0 < j) minNeighbor = std::min(minNeighbor, dp[i][j - 1]);

                dp[i][j] = 1 + minNeighbor;
            }
        }
        
        // The second pass: Pretend that we can only move up or left. 
        for (int i = m - 1; 0 <= i; --i)
        {
            for (int j = n - 1; 0 <= j; --j)
            {
                if (!dp[i][j]) continue;

                int minNeighbor = m * n;
                if (i < m - 1) minNeighbor = std::min(minNeighbor, dp[i + 1][j]);
                if (j < n - 1) minNeighbor = std::min(minNeighbor, dp[i][j + 1]);

                dp[i][j] = std::min(dp[i][j], 1 + minNeighbor);
            }
        }

        return dp;
    }
};