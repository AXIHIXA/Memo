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
    struct Edge
    {
        int s = 0;
        int t = 0;
        int w = 0;
    };

    void input()
    {
        std::scanf("%d %d\n", &n, &m);
        std::iota(root.begin(), root.end(), 0);

        for (int i = 0, s, t, w; i < m; ++i)
        {
            std::scanf("%d %d %d\n", &s, &t, &w);
            e[i] = {s, t, w};
        }
    }

    void solve()
    {
        std::sort(e.begin(), e.begin() + m, [](const Edge & a, const Edge & b) -> bool
        {
            return a.w < b.w;
        });

        for (int i = 0; i < m; ++i)
        {
            if (!connected(e[i].s, e[i].t))
            {
                unite(e[i].s, e[i].t);
                ans = std::max(ans, e[i].w);
            }
        }
    }

    void output() const
    {
        std::printf("%d %d\n", n - 1, ans);
    }

private:
    int find(int x)
    {
        if (x == root[x]) return x;
        return root[x] = find(root[x]);
    }

    void unite(int x, int y)
    {
        int rx = find(x), ry = find(y);
        if (rx == ry) return;
        root[rx] = ry;
    }

    bool connected(int x, int y)
    {
        return find(x) == find(y);
    }

private:
    static constexpr int kMaxN = 310;
    static constexpr int kMaxM = 8'010;

    int n = 0;
    int m = 0;

    std::array<Edge, kMaxM> e = {};
    std::array<int, kMaxN> root = {0};

    int ans = 0;
};


int main(int argc, char * argv[])
{
    Solution s;

    return EXIT_SUCCESS;
}