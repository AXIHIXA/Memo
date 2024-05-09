class Solution
{
public:
    int numDecodings(std::string s)
    {
        auto n = static_cast<const int>(s.size());

        std::vector<long long> dp(n + 1, 0);
        dp[n] = 1;

        if (s.back() == '0')
        {
            dp[n - 1] = 0;
        }
        else if (s.back() == '*')
        {
            dp[n - 1] = 9;
        }
        else
        {
            dp[n - 1] = 1;
        }

        for (int i = n - 2; 0 <= i; --i)
        {
            if (s[i] == '0')
            {
                continue;
            }

            if (s[i] == '*')
            {
                dp[i] = (9 * dp[i + 1]) % p;

                if (s[i + 1] == '*')
                {
                    dp[i] = (dp[i] + (15 * dp[i + 2]) % p) % p;
                }
                else if (s[i + 1] < '7')
                {
                    dp[i] = (dp[i] + (2 * dp[i + 2]) % p) % p;
                }
                else
                {
                    dp[i] = (dp[i] + dp[i + 2]) % p;
                }
            }
            else
            {
                dp[i] = dp[i + 1];

                if (s[i] < '3')
                {
                    if (s[i + 1] == '*')
                    {
                        if (s[i] == '1')
                        {
                            dp[i] = (dp[i] + (9 * dp[i + 2]) % p) % p;
                        }
                        else
                        {
                            dp[i] = (dp[i] + (6 * dp[i + 2]) % p) % p;
                        }
                    }
                    else if (s[i + 1] < '7')
                    {
                        dp[i] = (dp[i] + dp[i + 2]) % p;
                    }
                    else
                    {
                        if (s[i] == '1')
                        {
                            dp[i] = (dp[i] + dp[i + 2]) % p;
                        }
                    }
                }
            }
        }

        return dp[0];
    }

private:
    static constexpr int p = 1'000'000'007;
};