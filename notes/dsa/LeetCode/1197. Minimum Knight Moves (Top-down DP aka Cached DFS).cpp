class Solution 
{
public:
    int minKnightMoves(int x, int y) 
    {
        x = abs(x);
        y = abs(y);
        
        // Top-down DP (cached recursive DFS). 
        // Only consider the 1st-quadrant due to symmetry. 
        // dp[x][y] denotes the min #steps to reach (0, 0) from (x, y). 
        // dp[x][y] = min(dp[|x - 1|][|y - 2|], dp[|x - 2|][|y - 1|]) + 1. 
        dp.assign(x + 10, std::vector<int>(y + 10, -1));
        dp[0][0] = 0;
        dp[0][2] = 2;
        dp[1][1] = 2;
        dp[2][0] = 2;

        return dfs(x, y);
    }

private:
    int dfs(int x, int y)
    {
        if (dp[x][y] != -1)
        {
            return dp[x][y];
        }

        return dp[x][y] = min(dfs(abs(x - 1), abs(y - 2)), 
                              dfs(abs(x - 2), abs(y - 1))) + 1;
    }

    std::vector<std::vector<int>> dp;
};
