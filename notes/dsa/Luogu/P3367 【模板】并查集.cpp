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
        std::iota(root.begin(), root.begin() + n, 0);

        for (int i = 0, x, y, z; i < d; ++i)
        {
            std::scanf("%d %d %d\n", &z, &x, &y);

            if (z == 1)
            {
                unite(x, y);
            }
            else
            {
                if (find(x) == find(y))
                {
                    ans += 'Y';
                    ans += '\n';
                }
                else
                {
                    ans += 'N';
                    ans += '\n';
                }
            }
        }
    }

    void solve()
    {

    }

    void output() const
    {
        std::printf("%s\n", ans.c_str());
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

        if (rank[rx] < rank[ry]) root[rx] = ry;
        else if (rank[ry] < rank[rx]) root[ry] = rx;
        else root[ry] = rx, ++rank[rx];
    }

private:
    static constexpr int kSize = 10'010;

    int n = 0;
    int d = 0;

    std::array<int, kSize> root = {0};
    std::array<int, kSize> rank = {0};

    std::string ans;
};


int main(int argc, char * argv[])
{
    Solution s;

    return EXIT_SUCCESS;
}