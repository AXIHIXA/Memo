class Dp3d
{
public:
    int findMaxForm(std::vector<std::string> & strs, int m, int n)
    {
        auto len = static_cast<const int>(strs.size());

        std::vector a(len + 10, std::array<int, 2> {0, 0});

        for (int i = 0; i < len; ++i)
        {
            for (char c : strs[i])
            {
                ++a[i][c - '0'];
            }
        }

        std::vector dp(len + 10, std::vector(m + 10, std::vector<int>(n + 10, 0)));

        for (int i = 1; i <= len; ++i)
        {
            for (int mi = 0; mi <= m; ++mi)
            {
                for (int ni = 0; ni <= n; ++ni)
                {
                    dp[i][mi][ni] = dp[i - 1][mi][ni];

                    if (a[i - 1][0] <= mi && a[i - 1][1] <= ni)
                    {
                        dp[i][mi][ni] = std::max(
                                dp[i][mi][ni], 
                                1 + dp[i - 1][mi - a[i - 1][0]][ni - a[i - 1][1]]
                        );
                    }
                }
            }
        }

        return dp[len][m][n];
    }
};

class Dp2d
{
public:
    int findMaxForm(std::vector<std::string> & strs, int m, int n)
    {
        std::vector dp(m + 10, std::vector<int>(n + 10, 0));

        for (const std::string & s : strs)
        {
            std::array<int, 2> a = {0, 0};
            
            for (char c : s)
            {
                ++a[c - '0'];
            }
            
            for (int mi = m; a[0] <= mi; --mi)
            {
                for (int ni = n; a[1] <= ni; --ni)
                {
                    dp[mi][ni] = std::max(
                            dp[mi][ni], 
                            1 + dp[mi - a[0]][ni - a[1]]
                    );
                }
            }
        }

        return dp[m][n];
    }
};

// using Solution = Dp3d;
using Solution = Dp2d;
