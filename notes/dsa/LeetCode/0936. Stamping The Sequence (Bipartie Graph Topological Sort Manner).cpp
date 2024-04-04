class Solution
{
public:
    std::vector<int> movesToStamp(std::string stamp, std::string target)
    {
        auto stampSize = static_cast<const int>(stamp.size());
        auto m = static_cast<const int>(target.size());
        const int n = m - stampSize + 1;
        
        // Adjacency List of bipartie graph {target indices} -> {stampable indices}. 
        // g[i] = {a, b, c, ...} indicates that
        // Target[i] could be invalidated by stamping at positions a, b, c, ...
        std::vector g(m, std::vector<int>());
        std::vector inDegree(n, stampSize);
        std::queue<int> que;

        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < stampSize; ++j)
            {
                if (stamp[j] != target[i + j])
                {
                    // Target[i + j] is invalidated by stamping at position i. 
                    g[i + j].emplace_back(i);
                }
                else
                {
                    // The last stamp should invalidate nowhere. 
                    if (--inDegree[i] == 0)
                    {
                        que.emplace(i);
                    }
                }
            }
        }
        
        std::vector<int> ans;
        ans.reserve(n);

        std::vector<unsigned char> corrected(m, false);

        while (!que.empty())
        {
            int i = que.front();
            que.pop();
            ans.emplace_back(i);

            for (int j = 0; j < stampSize; ++j)
            {
                if (!corrected[i + j])
                {
                    corrected[i + j] = true;

                    for (int t : g[i + j])
                    {
                        if (--inDegree[t] == 0)
                        {
                            que.emplace(t);
                        }
                    }
                }
            }
        }

        if (ans.size() != n)
        {
            ans.clear();
        }

        std::reverse(ans.begin(), ans.end());
        return ans;
    }
};