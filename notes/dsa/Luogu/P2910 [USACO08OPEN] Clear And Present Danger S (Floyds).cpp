#include <bits/stdc++.h>


class Solution
{
public:
    Solution()
    {
        static const int _ = iosInit();
        input();
        solve();
        output();
    }

private:
    static int iosInit()
    {
        std::ios_base::sync_with_stdio(false);
        std::cin.tie(nullptr);
        std::cout.tie(nullptr);
        std::setvbuf(stdin, nullptr, _IOFBF, 1 << 20);
        std::setvbuf(stdout, nullptr, _IOFBF, 1 << 20);
        return 0;
    }

    void input()
    {
        std::scanf("%d %d\n", &n, &m);

        for (int i = 0; i < m; ++i)
        {
            std::scanf("%d\n", &route[i]);
        }

        for (auto & line : dist)
        {
            std::fill(line.begin(), line.end(), std::numeric_limits<int>::max());
        }

        for (int i = 1; i <= n; ++i)
        {
            for (int j = 1; j <= n; ++j)
            {
                std::scanf("%d", &dist[i][j]);
            }
        }
    }

    void solve()
    {
        for (int k = 1; k <= n; ++k)
        {
            for (int i = 1; i <= n; ++i)
            {
                for (int j = 1; j <= n; ++j)
                {
                    if (dist[i][k] < std::numeric_limits<int>::max() &&
                        dist[k][j] < std::numeric_limits<int>::max() &&
                        dist[i][k] + dist[k][j] < dist[i][j])
                    {
                        dist[i][j] = dist[i][k] + dist[k][j];
                    }
                }
            }
        }

        for (int i = 1; i < m; ++i)
        {
            ans += dist[route[i - 1]][route[i]];
        }
    }

    void output() const
    {
        std::printf("%d\n", ans);
    }

private:
    static constexpr int kMaxN = 110;
    static constexpr int kMaxM = 10'010;

    static std::array<std::array<int, kMaxN>, kMaxN> g;
    static std::array<std::array<int, kMaxN>, kMaxN> dist;
    static std::array<int, kMaxM> route;

    int n = 0;
    int m = 0;
    int ans = 0;
};


std::array<std::array<int, Solution::kMaxN>, Solution::kMaxN> Solution::dist = {};
std::array<int, Solution::kMaxM> Solution::route = {};


int main(int argc, char * argv[])
{
    Solution s;

    return EXIT_SUCCESS;
}