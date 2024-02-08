class Solution
{
public:
    int numSquares(int n)
    {
        int maxPfIdx = static_cast<int>(std::sqrt(n)) + 1;
        std::vector<int> pf;
        pf.reserve(maxPfIdx);

        for (int i = 1; i < maxPfIdx; ++i)
        {
            pf[i] = i * i;
        }

        std::vector<int> dp(n + 1, kIntMax);
        dp[0] = 0;

        for (int i = 1; i <= n; ++i)
        {
            for (int j = 1; j < maxPfIdx && pf[j] <= i; ++j)
            {
                dp[i] = std::min(dp[i], dp[i - pf[j]] + 1);
            }
        }

        return dp[n];
    }

private:
    static constexpr int kIntMax = std::numeric_limits<int>::max();
};