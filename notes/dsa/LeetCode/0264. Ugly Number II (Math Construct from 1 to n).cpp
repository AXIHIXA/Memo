class Solution
{
public:
    int nthUglyNumber(int n)
    {
        std::vector<int> dp(n + 1);
        dp[1] = 1;

        for (int i = 2, i2 = 1, i3 = 1, i5 = 1, a, b, c, mi; i <= n; ++i)
        {
            a = dp[i2] * 2;
            b = dp[i3] * 3;
            c = dp[i5] * 5;

            mi = std::min({a, b, c});
            
            // Note mi could equal to multiple in {a, b, c}!
            // Three ifs must be as-is to de-duplicate!
            if (mi == a) ++i2;
            if (mi == b) ++i3;
            if (mi == c) ++i5;

            dp[i] = mi;
        }

        return dp[n];
    }
};