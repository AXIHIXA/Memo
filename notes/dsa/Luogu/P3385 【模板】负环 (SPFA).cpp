#include <bits/stdc++.h>


class Solution
{
public:
    Solution()
    {
        static const int _ = iosInit();

        int t = 0;
        std::scanf("%d\n", &t);

        for (int i = 0; i < t; ++i)
        {
            clear();
            input();
            solve();
            output();
        }
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

    static void clear()
    {
        cnt = 0;
        std::fill(head.begin(), head.end(), 0);
        std::fill(to.begin(), to.end(), 0);
        std::fill(next.begin(), next.end(), 0);
        std::fill(weight.begin(), weight.end(), 0);

        std::fill(dist.begin(), dist.end(), std::numeric_limits<int>::max());

        ql = qr = 0;
        std::fill(inQueue.begin(), inQueue.end(), false);
        std::fill(updateCount.begin(), updateCount.end(), 0);

        // Reset ANSWER for multiple-testcase problems!!!
        hasNegativeLoop = false;
    }

    static void addEdge(int s, int t, int w)
    {
        ++cnt;
        next[cnt] = head[s];
        head[s] = cnt;
        to[cnt] = t;
        weight[cnt] = w;
    }

    static void input()
    {
        std::scanf("%d %d\n", &n, &m);

        for (int i = 0, s, t, w; i < m; ++i)
        {
            std::scanf("%d %d %d\n", &s, &t, &w);

            addEdge(s, t, w);

            if (0 <= w)
            {
                addEdge(t, s, w);
            }
        }
    }

    static void solve()
    {
        // Bellman-Ford AND SPFA.
        dist[1] = 0;
        ++updateCount[1];
        queue[qr++] = 1;
        inQueue[1] = true;

        while (ql < qr)
        {
            int s = queue[ql++];
            inQueue[s] = false;

            for (int e = head[s]; e != 0; e = next[e])
            {
                int t = to[e];
                int w = weight[e];

                if (static_cast<long long>(dist[s]) + w < dist[t])
                {
                    dist[t] = dist[s] + w;

                    if (!inQueue[t])
                    {
						++updateCount[t];

                        if (updateCount[t] == n + 1)
                        {
							hasNegativeLoop = true;
                            return;
						}

						queue[qr++] = t;
						inQueue[t] = true;
					}
                }
            }
        }
    }

    static void output()
    {
        hasNegativeLoop ? std::puts("YES") : std::puts("NO");
    }

private:
    static constexpr int kMaxN = 2'010;
    static constexpr int kMaxM = 6'010;  // <= 3000 undirected edges.
    static constexpr int kMaxQueueSize = 4'000'010;

    static int cnt;
    static std::array<int, kMaxN> head;
    static std::array<int, kMaxM> to;
    static std::array<int, kMaxM> next;
    static std::array<int, kMaxM> weight;

    static std::array<int, kMaxN> dist;

    static int ql;
    static int qr;
    static std::array<int, kMaxQueueSize> queue;
    static std::array<bool, kMaxN> inQueue;
    static std::array<int, kMaxN> updateCount;

    static int n;
    static int m;
    static bool hasNegativeLoop;
};


int Solution::cnt = 0;
std::array<int, Solution::kMaxN> Solution::head = {};
std::array<int, Solution::kMaxM> Solution::to = {};
std::array<int, Solution::kMaxM> Solution::next = {};
std::array<int, Solution::kMaxM> Solution::weight = {};

std::array<int, Solution::kMaxN> Solution::dist = {};

int Solution::ql = 0;
int Solution::qr = 0;
std::array<int, Solution::kMaxQueueSize> Solution::queue = {};
std::array<bool, Solution::kMaxN> Solution::inQueue = {};
std::array<int, Solution::kMaxN> Solution::updateCount = {};

int Solution::n = 0;
int Solution::m = 0;
bool Solution::hasNegativeLoop = false;


int main(int argc, char * argv[])
{
    std::freopen("var/1.txt", "r", stdin);
    Solution s;

    return EXIT_SUCCESS;
}