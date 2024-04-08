#include <bits/stdc++.h>


class Kruskal
{
public:
    struct Edge
    {
        int source = 0;
        int target = 0;
        int weight = 0;
    };

public:
    Kruskal()
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

        for (int i = 0, s, t, w; i < m; ++i)
        {
            std::scanf("%d %d %d\n", &s, &t, &w);
            e[i] = {s, t, w};
        }
    }

    void solve()
    {
        std::iota(root.begin(), root.end(), 0);

        std::sort(e.begin(), e.begin() + m, [](const auto & a, const auto & b)
        {
            return a.weight < b.weight;
        });

        for (int i = 0; i < m && cnt < n - 1; ++i)
        {
            auto [s, t, w] = e[i];

            if (!connected(s, t))
            {
                unite(s, t);
                ans += w;
                ++cnt;
            }
        }
    }

    void output() const
    {
        cnt == n - 1 ? std::printf("%d\n", ans) : std::puts("orz\n");
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

    bool connected(int x, int y)
    {
        return find(x) == find(y);
    }

private:
    static constexpr int kMaxN = 5'010;
    static constexpr int kMaxM = 200'010;

    int n = 0;
    int m = 0;

    std::array<Edge, kMaxM> e;

    std::array<int, kMaxN> root = {0};
    std::array<int, kMaxN> rank = {0};

    int ans = 0;
    int cnt = 0;
};


class Prim
{
public:
    Prim()
    {
        static const int _ = iosInit();
        input();
        solve();
        output();
    }

private:
    using VertIdx = int;
    using EdgeIdx = int;
    using Weight = int;

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
        std::scanf("%d %d\n", &n, &m);

        for (int i = 0, s, t, w; i < m; ++i)
        {
            std::scanf("%d %d %d\n", &s, &t, &w);
            addEdge(s, t, w);
            addEdge(t, s, w);
        }
    }

    void solve()
    {
        // 1节点出发
        std::fill(where.begin(), where.end(), -1);
		mstNodeCount = 1;
		where[1] = -2;

		for (int e = head[1]; e != 0; e = next[e])
        {
			addOrUpdateOrIgnore(e);
		}

		while (heapSize != 0)
        {
			auto [v, w] = pop();
			ans += w;

			for (int e = head[v]; e != 0; e = next[e])
            {
				addOrUpdateOrIgnore(e);
			}
		}
    }

    void output() const
    {
        mstNodeCount == n ? std::printf("%d\n", ans) : std::puts("orz\n");
    }

    void addEdge(VertIdx s, VertIdx t, Weight w)
    {
        ++cnt;
        next[cnt] = head[s];
        head[s] = cnt;
        to[cnt] = t;
        weight[cnt] = w;
    }

    // 当前处理的是编号为ei的边！
	void addOrUpdateOrIgnore(int e)
    {
		int t = to[e];
		int w = weight[e];

		// 去往v点，权重w
		if (where[t] == -1)
        {
			// v这个点，从来没有进入过堆！
			heap[heapSize] = {t, w};
			where[t] = heapSize++;
			push(where[t]);
		}
        else if (0 <= where[t])
        {
			// v这个点的记录，在堆上的位置是where[v]
			heap[where[t]].second = std::min(heap[where[t]].second, w);
			push(where[t]);
		}
	}

    void push(int i)
    {
        while (0 < i && heap[i].second < heap[(i - 1) >> 1].second)
        {
            swap(i, (i - 1) >> 1);
            i = (i - 1) >> 1;
        }
    }

    std::pair<VertIdx, Weight> pop()
    {
		auto [v, w] = heap[0];
		swap(0, --heapSize);
		heapify();
		where[v] = -2;
		++mstNodeCount;
        return {v, w};
    }

    void heapify()
    {
        for (int i = 0, l = 1; l < heapSize; )
        {
            int best = l + 1 < heapSize && heap[l + 1].second < heap[l].second ? l + 1 : l;
            best = heap[i].second < heap[best].second ? i : best;

            if (best == i)
            {
                break;
            }

            swap(i, best);
            i = best;
            l = (i << 1) + 1;
        }
    }

    void swap(int i, int j)
    {
        int a = heap[i].first;
		int b = heap[j].first;
		where[a] = j;
		where[b] = i;
		std::swap(heap[i], heap[j]);
    }

private:
    static constexpr int kMaxN = 5'010;
    static constexpr int kMaxM = 400'010;

    // Graph size.
    int n = 0;
    int m = 0;

    // Forward-star list.
    int cnt = 0;
    std::array<EdgeIdx, kMaxN> head = {0};
    std::array<VertIdx, kMaxM> to = {0};
    std::array<EdgeIdx, kMaxM> next = {0};
    std::array<Weight, kMaxM> weight = {0};

    // Customized heap.
    int heapSize = 0;
    std::array<std::pair<VertIdx, Weight>, kMaxN> heap = {};

    // where[vert] == ? means that vertex #vert is:
    // -1: unvisited;
    // -2: popped;
    // >0: in heap at index where[vert].
    std::array<int, kMaxN> where = {};

    // MST info.
    int ans = 0;
    int mstNodeCount = 0;
};


int main(int argc, char * argv[])
{
    // Kruskal k;
    Prim p;

    return EXIT_SUCCESS;
}