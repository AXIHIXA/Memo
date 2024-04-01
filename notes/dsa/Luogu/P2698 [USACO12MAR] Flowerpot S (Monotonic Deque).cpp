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

private:
    void input()
    {
        std::scanf("%d %d\n", &n, &d);

        for (int i = 0; i < n; ++i)
        {
            std::scanf("%d %d\n", &xy[i].first, &xy[i].second);
        }
    }

    void solve()
    {
        std::sort(xy.begin(), xy.begin() + n, [](const auto & a, const auto & b)
        {
            return a.first < b.first;
        });

        int maxDl = 0, maxDr = 0;
        int minDl = 0, minDr = 0;

        // Min-length sliding window s.t. d <= max - min.
        for (int ll = 0, rr = 0; rr < n; ++rr)
        {
            while (maxDl < maxDr && xy[maxDeq[maxDr - 1]].second <= xy[rr].second)
            {
                --maxDr;
            }

            maxDeq[maxDr++] = rr;

            while (minDl < minDr && xy[rr].second <= xy[minDeq[minDr - 1]].second)
            {
                --minDr;
            }

            minDeq[minDr++] = rr;

            while (ll <= rr && d <= xy[maxDeq[maxDl]].second - xy[minDeq[minDl]].second)
            {
                ans = std::min(ans, xy[rr].first - xy[ll].first);

                ++ll;

                while (maxDl < maxDr && maxDeq[maxDl] < ll)
                {
                    ++maxDl;
                }

                while (minDl < minDr && minDeq[minDl] < ll)
                {
                    ++minDl;
                }
            }
        }
    }

    void output() const
    {
        std::printf("%d\n", ans == std::numeric_limits<int>::max() ? -1 : ans);
    }

private:
    static constexpr int kSize = 100'010;

    static std::array<std::pair<int, int>, kSize> xy;
    static std::array<int, kSize> maxDeq;
    static std::array<int, kSize> minDeq;

private:
    int n = 0;
    int d = 0;

    int ans = std::numeric_limits<int>::max();
};


std::array<std::pair<int, int>, Solution::kSize> Solution::xy;
std::array<int, Solution::kSize> Solution::maxDeq;
std::array<int, Solution::kSize> Solution::minDeq;


int main(int argc, char * argv[])
{
    Solution s;

    return EXIT_SUCCESS;
}
