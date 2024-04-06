class Solution
{
public:
    std::vector<int> loudAndRich(std::vector<std::vector<int>> & richer, std::vector<int> & quiet)
    {
        auto n = static_cast<const int>(quiet.size());
        auto m = static_cast<const int>(richer.size());

        std::vector g(n, std::vector<int>());
        std::vector<int> inDegree(n, 0);

        for (const auto & e : richer)
        {
            int s = e[0], t = e[1];
            g[s].emplace_back(t);
            ++inDegree[t];
        }

        std::vector<int> ans(n);
        std::iota(ans.begin(), ans.end(), 0);

        std::queue<int> que;

        for (int v = 0; v < n; ++v)
        {
            if (inDegree[v] == 0)
            {
                que.emplace(v);
            }
        }

        while (!que.empty())
        {
            int s = que.front();
            que.pop();

            for (int t : g[s])
            {
                if (quiet[ans[s]] < quiet[ans[t]])
                {
                    ans[t] = ans[s];
                }

                if (--inDegree[t] == 0)
                {
                    que.emplace(t);
                }
            }
        }

        return ans;
    }
};