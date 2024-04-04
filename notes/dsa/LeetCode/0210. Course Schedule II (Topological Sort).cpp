class Solution
{
public:
    std::vector<int> findOrder(int numCourses, std::vector<std::vector<int>> & prerequisites)
    {
        const int n = numCourses;
        auto m = static_cast<const int>(prerequisites.size());

        // Adjacency List. 
        std::vector g(n, std::vector<int>());
        std::vector<int> inDegree(n, 0);

        for (const auto & e : prerequisites)
        {
            g[e[1]].emplace_back(e[0]);
            ++inDegree[e[0]];
        }

        std::queue<int> que;

        for (int i = 0; i < n; ++i)
        {
            if (inDegree[i] == 0)
            {
                que.emplace(i);
            }
        }

        std::vector<int> ans;
        ans.reserve(n);

        while (!que.empty())
        {
            int s = que.front();
            que.pop();
            ans.emplace_back(s);

            for (int t : g[s])
            {
                if (--inDegree[t] == 0)
                {
                    que.emplace(t);
                }
            }
        }

        // In case no topo sort valid. 
        if (ans.size() < n)
        {
            ans.clear();
        }

        return ans;
    }
};