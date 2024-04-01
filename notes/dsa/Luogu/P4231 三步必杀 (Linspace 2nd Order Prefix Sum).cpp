#include <bits/stdc++.h>


class Solution
{
public:
    Solution()
    {
        static const int _ = init();
        input();
        solve();
        output();
    }

    void input()
    {
        int m, l, r, s, e;
        std::scanf("%d %d\n", &n, &m);

        for (int i = 0; i < m; ++i)
        {
            std::scanf("%d %d %d %d\n", &l, &r, &s, &e);
            add(l, r, s, e);
        }
    }

    void solve()
    {
        for (int i = 1; i <= n; ++i) ps[i] += ps[i - 1];
        for (int i = 1; i <= n; ++i) ps[i] += ps[i - 1];
    }

    void output()
    {
        long long xorSum = 0;
        for (int i = 1; i <= n; ++i) xorSum ^= ps[i];
        long long maxi = *std::max_element(ps.cbegin() + 1, ps.cbegin() + n + 1);
        std::printf("%lld %lld\n", xorSum, maxi);
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

    static void add(int l, int r, int s, int e)
    {
        int d = (e - s) / (r - l);
        ps[l] += s;
        ps[l + 1] += d - s;
        ps[r + 1] -= d + e;
        ps[r + 2] += e;
    }

private:
    static constexpr int kSize = 10'000'010;
    static std::array<long long, kSize> ps;

    int n = 0;
};


std::array<long long, Solution::kSize> Solution::ps = {0};


int main(int argc, char * argv[])
{
    Solution s;

    return EXIT_SUCCESS;
}
