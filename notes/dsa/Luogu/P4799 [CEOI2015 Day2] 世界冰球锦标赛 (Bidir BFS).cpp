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
        int e1 = f(0, n >> 1, 0, m, ls.data(), 0);
        int e2 = f(n >> 1, n, 0, m, rs.data(), 0);

        std::sort(ls.begin(), ls.begin() + e1);
        std::sort(rs.begin(), rs.begin() + e2);

        for (int i = e1 - 1, j = 0; 0 <= i; --i)
        {
			while (j < e2 && ls[i] + rs[j] <= m)
            {
				++j;
			}

			ans += j;
		}
    }

    void output() const
    {
        std::printf("%lld\n", ans);
    }

    int f(int b, int e, long long s, long long w, long long * res, int j)
    {
        if (w < s)
        {
			return j;
		}

		// s <= w
		if (b == e)
        {
			res[j++] = s;
		}
        else
        {
			// 不要arr[i]位置的数
			j = f(b + 1, e, s, w, res, j);

			// 要arr[i]位置的数
			j = f(b + 1, e, s + arr[b], w, res, j);
		}

		return j;
    }

private:
    static constexpr int kMaxN = 50;
    static constexpr int kMaxM = 1 << 20;

    static std::array<long long, kMaxN> arr;
    static std::array<long long, kMaxM> ls;
    static std::array<long long, kMaxM> rs;

    int n = 0;
    long long m = 0LL;

    long long ans = 0LL;
};


std::array<long long, Solution::kMaxN> Solution::arr = {};
std::array<long long, Solution::kMaxM> Solution::ls = {};
std::array<long long, Solution::kMaxM> Solution::rs = {};


int main(int argc, char * argv[])
{
    Solution s;

    return EXIT_SUCCESS;
}