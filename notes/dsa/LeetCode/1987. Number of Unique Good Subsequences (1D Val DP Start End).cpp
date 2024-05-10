class Solution
{
public:
    int numberOfUniqueGoodSubsequences(std::string binary)
    {
        // Number of unique subseqs in forms of: 
        // dp[0]  0...0
        // dp[1]  0...1
        // dp[2]  1...0
        // dp[3]  1...1
        std::array<int, 4> dp = {0, 0, 0, 0};

        for (char c : binary)
        {
            if (c == '0')
            {
                dp[0] = 1;
                dp[2] = (dp[2] + dp[3]) % p;
            }
            else
            {
                // +1 because dp[2] + dp[3] omits seq "1". 
                dp[3] = (dp[2] + dp[3] + 1) % p;
            }
        }

        int ans = 0;

        for (int x : dp)
        {
            ans = (ans + x) % p;
        }

        return ans;
    }

private:
    static constexpr int p = 1'000'000'007;
};