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

        heapSize = 0;
        std::fill(dist.begin(), dist.end(), std::numeric_limits<int>::max());
        std::fill(where.begin(), where.end(), -1);
    }

    static void addEdge(int s, int t, int w)
    {
        ++cnt;
        next[cnt] = head[s];
        head[s] = cnt;
        to[cnt] = t;
        weight[cnt] = w;
    }

    static void updateHeap(int v, int c)
    {
        if (where[v] == -1)
        {
            heap[heapSize] = v;
            dist[v] = c;
            where[v] = heapSize++;
            pushHeap(where[v]);
        }
        else if (0 <= where[v])
        {
            dist[v] = std::min(dist[v], c);
            pushHeap(where[v]);
        }
    }

    static void popHeap()
    {
        swapHeap(0, --heapSize);
        heapify(0);
        where[heap[heapSize]] = -2;
    }

    static void swapHeap(int i, int j)
    {
        std::swap(heap[i], heap[j]);
        where[heap[i]] = i;
        where[heap[j]] = j;
    }

    static void pushHeap(int i)
    {
        while (dist[heap[i]] < dist[heap[(i - 1) / 2]])
        {
            swapHeap(i, (i - 1) / 2);
            i = (i - 1) / 2;
        }
    }

    static void heapify(int i)
    {
        for (int l = (i << 1) + 1; l < heapSize; )
        {
            int best = l + 1 < heapSize && dist[heap[l + 1]] < dist[heap[l]] ? l + 1 : l;
            best = dist[heap[best]] < dist[heap[i]] ? best : i;
            if (best == i) break;
            swapHeap(best, i);
            i = best;
            l = (i << 1) + 1;
        }
    }

private:
    void input()
    {
        clear();
        std::scanf("%d %d %d\n", &n, &m, &source);

        for (int i = 0, s, t, w; i < m; ++i)
        {
            std::scanf("%d %d %d\n", &s, &t, &w);
            addEdge(s, t, w);
        }
    }

    void solve()
    {
        dist[source] = 0;
        updateHeap(source, 0);

        while (heapSize != 0)
        {
            int s = heap.front();
            popHeap();

            for (int e = head[s]; e != 0; e = next[e])
            {
                updateHeap(to[e], dist[s] + weight[e]);
            }
        }
    }

    void output() const
    {
        for (int t = 1; t <= n; ++t)
        {
            std::printf("%d ", dist[t]);
        }

        std::printf("\n");
    }

private:
    static constexpr int kMaxN = 100'010;
    static constexpr int kMaxM = 200'010;

    // 链式前向星
    static int cnt;
    static std::array<int, kMaxN> head;
    static std::array<int, kMaxM> to;
    static std::array<int, kMaxM> next;
    static std::array<int, kMaxM> weight;

    // 反向索引堆
    static int heapSize;
    static std::array<int, kMaxN> heap;
    static std::array<int, kMaxN> dist;
    static std::array<int, kMaxN> where;

    int n = 0;
    int m = 0;
    int source = 0;
};


int Solution::cnt = 0;
std::array<int, Solution::kMaxN> Solution::head = {};
std::array<int, Solution::kMaxM> Solution::to = {};
std::array<int, Solution::kMaxM> Solution::next = {};
std::array<int, Solution::kMaxM> Solution::weight = {};

int Solution::heapSize = 0;
std::array<int, Solution::kMaxN> Solution::heap = {};
std::array<int, Solution::kMaxN> Solution::dist = {};
std::array<int, Solution::kMaxN> Solution::where = {};


int main(int argc, char * argv[])
{
    Solution s;

    return EXIT_SUCCESS;
}
