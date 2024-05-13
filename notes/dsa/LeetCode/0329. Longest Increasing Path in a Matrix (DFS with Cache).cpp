class Solution
{
public:
    int longestIncreasingPath(std::vector<std::vector<int>> & matrix)
    {
        auto m = static_cast<const int>(matrix.size());
        auto n = static_cast<const int>(matrix.front().size());

        for (auto & line : memo)
        {
            std::fill(line.begin(), line.end(), 0);
        }

        int ans = 1;

        for (int x = 0; x < m; ++x)
        {
            for (int y = 0; y < n; ++y)
            {
                ans = std::max(ans, dfs(matrix, m, n, x, y));
            }
        }

        return ans;
    }

private:
    static int dfs(
            const std::vector<std::vector<int>> & matrix, 
            int m, 
            int n, 
            int x, 
            int y)
    {
        if (0 < memo[x][y])
        {
            return memo[x][y];
        }

        int succLength = 0;

        for (int d = 0; d < 4; ++d)
        {
            int x1 = x + dx[d];
            int y1 = y + dy[d];
            
            if (0 <= x1 && x1 < m && 0 <= y1 && y1 < n && matrix[x][y] < matrix[x1][y1])
            {
                succLength = std::max(succLength, dfs(matrix, m, n, x1, y1));
            }
        }

        return memo[x][y] = 1 + succLength;
    }

private:
    static constexpr std::array<int, 4> dx = {1, 0, -1, 0};
    static constexpr std::array<int, 4> dy = {0, 1, 0, -1};

    static constexpr int kMaxSize = 201;
    
    static std::array<std::array<int, kMaxSize>, kMaxSize> memo;
};

std::array<std::array<int, Solution::kMaxSize>, Solution::kMaxSize> Solution::memo = {};
