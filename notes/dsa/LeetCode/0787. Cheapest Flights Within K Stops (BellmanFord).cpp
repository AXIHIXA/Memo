class LayeredGraphDijkstra
{
public:
    int findCheapestPrice(int n, std::vector<std::vector<int>> & flights, int src, int dst, int k)
    {
        std::vector g(n + 1, std::vector<std::pair<int, int>>());

        for (const auto & f : flights)
        {
            g[f[0]].emplace_back(f[1], f[2]);
        }

        auto cmp = 
        [](const std::tuple<int, int, int> & a, 
           const std::tuple<int, int, int> & b) -> bool
        {
            return std::get<2>(a) > std::get<2>(b);
        };

        std::priority_queue<
                std::tuple<int, int, int>, 
                std::vector<std::tuple<int, int, int>>, 
                decltype(cmp)
        > heap;

        heap.emplace(src, 0, 0);

        std::vector dist(n + 1, std::vector<int>(k + 2, std::numeric_limits<int>::max()));
        dist[src][0] = 0;

        int ans = std::numeric_limits<int>::max();

        while (!heap.empty())
        {
            auto [s, lv, d] = heap.top();
            heap.pop();

            if (s == dst)
            {
                ans = std::min(ans, dist[s][lv]);
            }

            if (lv <= k)
            {
                for (auto [t, w] : g[s])
                {
                    if (d + w < dist[t][lv + 1])
                    {
                        dist[t][lv + 1] = d + w;
                        heap.emplace(t, lv + 1, dist[t][lv + 1]);
                    }
                }
            }
        }

        return ans == std::numeric_limits<int>::max() ? -1 : ans;
    }
};

class BellmanFord
{
public:
    int findCheapestPrice(int n, std::vector<std::vector<int>> & flights, int src, int dst, int k)
    {
        std::vector<int> dist(n, std::numeric_limits<int>::max());
        dist[src] = 0;

        for (int i = 0; i <= k; ++i)
        {
            std::vector<int> tmp = dist;
            
            for (const auto & e : flights)
            {
                int s = e[0], t = e[1], w = e[2];

                if (static_cast<long long>(dist[s]) + w < tmp[t])
                {
                    tmp[t] = dist[s] + w;
                }
            }

            dist = std::move(tmp);
        }

        return dist[dst] == std::numeric_limits<int>::max() ? -1 : dist[dst];
    }
};

// using Solution = LayeredGraphDijkstra;
using Solution = BellmanFord;
