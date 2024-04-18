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
    static void clear()
    {
        cnt = 0;
        std::fill(head.begin(), head.end(), 0);
        std::fill(to.begin(), to.end(), 0);
        std::fill(next.begin(), next.end(), 0);

        for (auto & v : visited)
        {
            std::fill(v.begin(), v.end(), false);
        }

        for (auto & d : dist)
        {
            std::fill(d.begin(), d.end(), std::numeric_limits<int>::max());
        }
    }

    static void addEdge(int s, int t, int w)
    {
        ++cnt;
        next[cnt] = head[s];
        head[s] = cnt;
        to[cnt] = t;
        weight[cnt] = w;
    }

private:
    void input()
    {
        clear();
        std::scanf("%d %d %d\n", &n, &m, &k);
        std::scanf("%d %d\n", &source, &target);
        ++source, ++target;  // 0-indexed to 1-indexed.

        for (int i = 0, s, t, w; i < m; ++i)
        {
            std::scanf("%d %d %d\n", &s, &t, &w);
            ++s, ++t;  // 0-indexed to 1-indexed.
            addEdge(s, t, w);
            addEdge(t, s, w);
        }
    }

    void solve()
    {
        auto gt2 =
        [](const std::tuple<int, int, int> & a, const std::tuple<int, int, int> & b) -> bool
        {
            return std::get<2>(a) > std::get<2>(b);
        };

        std::priority_queue<
                std::tuple<int, int, int>,
                std::vector<std::tuple<int, int, int>>,
                decltype(gt2)
        > heap;

        dist[source][0] = 0;
        heap.emplace(source, 0, 0);

        while (!heap.empty())
        {
            auto [s, sUsed, sDist] = heap.top();
            heap.pop();
//            std::printf("Pop  %d %d %d\n", s, sUsed, sDist);

            if (s == target)
            {
                ans = sDist;
                return;
            }

            if (visited[s][sUsed])
            {
                continue;
            }

            visited[s][sUsed] = true;

            for (int e = head[s]; e != 0; e = next[e])
            {
                int t = to[e];
                int w = weight[e];

                int tUsed = sUsed;
                int tDist = sDist + w;

                if (!visited[t][tUsed] && tDist < dist[t][tUsed])
                {
                    dist[t][tUsed] = tDist;
                    heap.emplace(t, tUsed, tDist);
//                    std::printf("Push %d %d %d\n", t, tUsed, tDist);
                }

                if (sUsed < k)
                {
                    tUsed = sUsed + 1;
                    tDist = sDist;

                    if (!visited[t][tUsed] && tDist < dist[t][tUsed])
                    {
                        dist[t][tUsed] = tDist;
                        heap.emplace(t, tUsed, tDist);
//                        std::printf("Push %d %d %d\n", t, tUsed, tDist);
                    }
                }
            }
        }
    }

    void output() const
    {
        std::printf("%d\n", ans);
    }

private:
    static constexpr int kMaxN = 10'010;
    static constexpr int kMaxM = 100'010;  // 5e4 undirected edges.
    static constexpr int kMaxK = 20;

    // Foward-star list.
    static int cnt;
    static std::array<int, kMaxN> head;
    static std::array<int, kMaxM> to;
    static std::array<int, kMaxM> next;
    static std::array<int, kMaxM> weight;

    // Dijkstra utilities.
    static std::array<std::array<bool, kMaxK>, kMaxN> visited;
    static std::array<std::array<int, kMaxK>, kMaxN> dist;

    int n = 0;
    int m = 0;
    int k = 0;
    int source = 0;
    int target = 0;
    int ans = -1;
};


int Solution::cnt = 0;
std::array<int, Solution::kMaxN> Solution::head = {};
std::array<int, Solution::kMaxM> Solution::to = {};
std::array<int, Solution::kMaxM> Solution::next = {};
std::array<int, Solution::kMaxM> Solution::weight = {};

std::array<std::array<bool, Solution::kMaxK>, Solution::kMaxN> Solution::visited = {};
std::array<std::array<int, Solution::kMaxK>, Solution::kMaxN> Solution::dist = {};


int main(int argc, char * argv[])
{
    Solution s;

    return EXIT_SUCCESS;
}