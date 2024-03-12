class Solution
{
public:
    std::vector<int> countBits(int n)
    {
        std::vector<int> dp(n + 1);
        dp[0] = 0;

        // 1. 
        // x & (x - 1) : x's least significant bit unset. 
        //               #1s in bin(x) == #1s in bin(x & (x - 1)) + 1. 
        // 
        // 2. 
        // (x >> 1)    : #1s in bin(x) == #1s in bin(x >> 1) + whether x's last bit is 1. 

        for (int x = 1; x <= n; ++x)
        {
            // dp[x] = 1 + dp[x & (x - 1)];
            dp[x] = dp[x >> 1] + (x & 1);
        }

        return dp;
    }
};
