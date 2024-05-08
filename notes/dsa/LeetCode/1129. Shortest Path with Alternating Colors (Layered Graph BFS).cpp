class Solution
{
public:
    std::vector<int> 
    shortestAlternatingPaths(
            int n, 
            std::vector<std::vector<int>> & redEdges, 
            std::vector<std::vector<int>> & blueEdges
    ) 
    {
        std::array<std::vector<std::vector<int>>, 2> g = {
                std::vector<std::vector<int>>(n, std::vector<int>()), 
                std::vector<std::vector<int>>(n, std::vector<int>())
        };

        for (const auto & e : redEdges)
        {
            g[0][e[0]].emplace_back(e[1]);
        }

        for (const auto & e : blueEdges)
        {
            g[1][e[0]].emplace_back(e[1]);
        }

        std::vector dist(
                n, 
                std::array<int, 2> {   
                        std::numeric_limits<int>::max(), 
                        std::numeric_limits<int>::max()
                }
        );
        
        dist[0][0] = 0;
        dist[0][1] = 0;

        std::queue<std::pair<int, int>> que;
        que.emplace(0, 0);
        que.emplace(0, 1);

        while (!que.empty())
        {
            auto levelSize = static_cast<const int>(que.size());

            for (int i = 0; i < levelSize; ++i)
            {
                auto [s, bs] = que.front();
                que.pop();

                bool bt = !bs;

                for (int t : g[bt][s])
                {
                    if (dist[s][bs] + 1 < dist[t][bt])
                    {
                        dist[t][bt] = dist[s][bs] + 1;
                        que.emplace(t, bt);
                    }
                }
            }
        }

        std::vector<int> ans(n);

        for (int i = 0; i < n; ++i)
        {
            ans[i] = std::min(dist[i][0], dist[i][1]);
            
            if (ans[i] == std::numeric_limits<int>::max())
            {
                ans[i] = -1;
            }
        }

        return ans;
    }
};