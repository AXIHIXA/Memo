#include <bits/stdc++.h>


class Solution
{
public:
    using VertIdx = int;
    using EdgeIdx = int;

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
        std::scanf("%d %d\n", &n, &m);

        for (int i = 0, s, t; i < m; ++i)
        {
            std::scanf("%d %d\n", &s, &t);
            addEdge(s, t);
            ++inDegree[t];
        }
    }

    void solve()
    {
        std::queue<VertIdx> que;

        for (VertIdx v = 1; v <= n; ++v)
        {
            if (inDegree[v] == 0)
            {
                que.emplace(v);
                lineCount[v] = 1;
            }
        }

        while (!que.empty())
        {
            VertIdx s = que.front();
            que.pop();

            if (head[s] == 0)
            {
                ans = (ans + lineCount[s]) % kMod;
                continue;
            }

            for (EdgeIdx e = head[s]; e != 0; e = next[e])
            {
                VertIdx t = to[e];
                lineCount[t] = (lineCount[t] + lineCount[s]) % kMod;

                if (--inDegree[t] == 0)
                {
                    que.emplace(t);
                }
            }
        }
    }

    void output() const
    {
        std::printf("%d\n", ans);
    }

private:
    void addEdge(VertIdx s, VertIdx t)
    {
        ++cnt;
        next[cnt] = head[s];
        head[s] = cnt;
        to[cnt] = t;
    }

private:
    static constexpr int kMod = 80'112'002;
    static constexpr int kMaxN = 5'010;
    static constexpr int kMaxM = 500'010;

    int n = 0;
    int m = 0;

    int cnt = 0;
    std::array<EdgeIdx, kMaxN> head = {0};
    std::array<EdgeIdx, kMaxM> next = {0};
    std::array<VertIdx, kMaxM> to = {0};

    std::array<int, kMaxN> inDegree = {0};
    std::array<int, kMaxN> lineCount = {0};

    int ans = 0;
};


int main(int argc, char * argv[])
{
    Solution s;

    return EXIT_SUCCESS;
}
