class Solution
{
public:
    int maxSumAfterPartitioning(std::vector<int> & arr, int k)
    {
        int n = arr.size();
        if (n == 1) return arr[0];

        std::vector<int> dp(n + 1, 0);
        
        for (int i = n - 1; 0 <= i; --i)
        {
            for (int j = i, rr = std::min(n, i + k), maximum = 0; j < rr; ++j)
            {
                maximum = std::max(maximum, arr[j]);
                dp[i] = std::max(dp[i], dp[j + 1] + maximum * (j - i + 1));
            }
        }

        return dp[0];
    }
};