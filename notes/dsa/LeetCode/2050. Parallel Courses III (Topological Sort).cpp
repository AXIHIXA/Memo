class Solution
{
public:
    int minimumTime(int n, std::vector<std::vector<int>> & relations, std::vector<int> & time)
    {
        auto m = static_cast<const int>(relations.size());
        
        std::vector g(n + 1, std::vector<int>());
        std::vector<int> inDegree(n + 1, 0);
        std::vector<int> delay(n + 1, 0);
        time.emplace(time.begin(), 0);

        for (const auto & e : relations)
        {
            int s = e[0], t = e[1];
            g[s].emplace_back(t);
            ++inDegree[t];
        }

        std::queue<int> que;

        for (int v = 1; v <= n; ++v)
        {
            if (inDegree[v] == 0)
            {
                que.emplace(v);
            }
        }

        int ans = 0;

        while (!que.empty())
        {
            int s = que.front();
            que.pop();

            if (g[s].empty())
            {
                ans = std::max(ans, time[s]);
                continue;
            }

            for (int t : g[s])
            {
                delay[t] = std::max(delay[t], time[s]);

                if (--inDegree[t] == 0)
                {
                    time[t] += delay[t];
                    que.emplace(t);
                }
            }
        }

        return ans;
    }
};