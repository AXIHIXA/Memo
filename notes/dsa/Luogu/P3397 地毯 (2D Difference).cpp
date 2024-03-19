#include <bits/stdc++.h>

#include <fmt/core.h>
#include <fmt/ranges.h>


class Solution
{
public:
    Solution()
    {
        static const int _ = init();
    }

    void input()
    {
        int m;
        std::scanf("%d %d\n", &n, &m);

        for (int i = 0; i < m; ++i)
        {
            int a;
            int b;
            int c;
            int d;
            std::scanf("%d %d %d %d\n", &a, &b, &c, &d);
            add(a, b, c, d, 1);
        }
    }

    void solve()
    {
        for (int i = 1; i <= n; ++i)
        {
            for (int j = 1; j <= n; ++j)
            {
                diff[i][j] += diff[i - 1][j] + diff[i][j - 1] - diff[i - 1][j - 1];
            }
        }
    }

    void output()
    {
        for (int i = 1; i <= n; ++i)
        {
            for (int j = 1; j <= n; ++j)
            {
                std::printf("%d ", diff[i][j]);
            }

            std::printf("\n");
        }

        std::printf("\n");
    }

private:
    static int init()
    {
        std::ios_base::sync_with_stdio(false);
        std::cin.tie(nullptr);
        std::cout.tie(nullptr);
        std::setvbuf(stdin, nullptr, _IOFBF, 1 << 20);
        std::setvbuf(stdout, nullptr, _IOFBF, 1 << 20);
        return 0;
    }

private:
    void add(int a, int b, int c, int d, int k)
    {
        diff[a][b] += k;
        diff[c + 1][b] -= k;
        diff[a][d + 1] -= k;
        diff[c + 1][d + 1] += k;
    }

private:
    static constexpr int kSize = 1010;
    static std::array<std::array<int, kSize>, kSize> diff;

private:
    int n = 0;
};


std::array<std::array<int, Solution::kSize>, Solution::kSize> Solution::diff = {0};


int main(int argc, char * argv[])
{
    #ifdef __CLION_IDE__
    std::freopen("var/1.txt", "r", stdin);
    std::freopen("var/2.txt", "w+", stdout);
    #endif

    Solution s;
    s.input();
    s.solve();
    s.output();

    return EXIT_SUCCESS;
}
