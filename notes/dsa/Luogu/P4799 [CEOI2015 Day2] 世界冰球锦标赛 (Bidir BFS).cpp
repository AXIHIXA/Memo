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
        std::scanf("%d %lld\n", &n, &m);

        for (int i = 0; i < n; ++i)
        {
            std::scanf("%lld", &arr[i]);
        }
    }

    void solve()
    {
        f(0, n >> 1, 0, ls.data(), &lsSize);
        f(n >> 1, n, 0, rs.data(), &rsSize);

        std::sort(ls.begin(), ls.begin() + lsSize);
        std::sort(rs.begin(), rs.begin() + rsSize);

        for (int i = 0, j = rsSize - 1; 0 <= j; --j)
        {
            while (i < lsSize && ls[i] + rs[j] <= m)
            {
                ++i;
            }

            ans += i;
        }
    }

    void output() const
    {
        std::printf("%lld\n", ans);
    }

    void f(int b, int e, long long cur, long long * res, int * size)
    {
        if (m < cur)
        {
            return;
        }

        if (b == e)
        {
            res[(*size)++] = cur;
        }
        else
        {
            f(b + 1, e, cur, res, size);
            f(b + 1, e, cur + arr[b], res, size);
        }
    }

private:
    static constexpr int kMaxN = 50;
    static constexpr int kMaxM = 1 << 20;

    static std::array<long long, kMaxN> arr;

    static int lsSize;
    static int rsSize;
    static std::array<long long, kMaxM> ls;
    static std::array<long long, kMaxM> rs;

    int n = 0;
    long long m = 0LL;

    long long ans = 0LL;
};


std::array<long long, Solution::kMaxN> Solution::arr = {};

int Solution::lsSize = 0;
int Solution::rsSize = 0;
std::array<long long, Solution::kMaxM> Solution::ls = {};
std::array<long long, Solution::kMaxM> Solution::rs = {};


int main(int argc, char * argv[])
{
    std::freopen("var/1.txt", "r", stdin);
    Solution s;

    return EXIT_SUCCESS;
}
