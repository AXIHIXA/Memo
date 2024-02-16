class Solution
{
public:
    int removeBoxes(std::vector<int> & boxes)
    {
        // dp[l][r][k]: 
        // Max points obtainable with subarray x[l..r], 
        // where k succeeding elements == x[r] (not including x[r]) are combined together. 
        // 
        // l ... i ...   r |<- k ->|
        // ? ... b ... a b b b ... b 
        // 
        // dp[l][r][k] = maximum of:
        //     Option 1: Combine x[r] ... x[r+k]: 
        //         dp[l][r-1][0] + (k+1)**2, 
        //     Option 2: Combine x[i] with x[r] ... x[r+k] 
        //               by removing x[i+1...r-1] in advance. 
        //         dp[l][i][k+1] + dp[i+1][r-1][0] for l <= i < r && boxes[i] == boxes[r]. 

        std::array<std::array<std::array<int, 100>, 100>, 100> dp {0};
        return maxPoints(boxes, dp, 0, boxes.size() - 1, 0);
    }

private:
    static int maxPoints(
            const std::vector<int> & boxes, 
            std::array<std::array<std::array<int, 100>, 100>, 100> & dp, 
            int l, 
            int r, 
            int k)
    {
        if (r < l) return 0;

        while (l < r && boxes[r - 1] == boxes[r]) --r, ++k;
        if (dp[l][r][k]) return dp[l][r][k];
        int ans = maxPoints(boxes, dp, l, r - 1, 0) + (k + 1) * (k + 1);

        for (int i = l; i < r; ++i)
        {
            if (boxes[i] == boxes[r])
            {
                ans = std::max(
                        ans, 
                        maxPoints(boxes, dp, l, i, k + 1) + maxPoints(boxes, dp, i + 1, r - 1, 0)
                );
            }
        }

        return dp[l][r][k] = ans;
    }
};