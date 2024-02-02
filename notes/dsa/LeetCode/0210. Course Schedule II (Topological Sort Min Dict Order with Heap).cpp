class Solution
{
public:
    std::vector<int> findOrder(int numCourses, std::vector<std::vector<int>> & prerequisites)
    {
        int n = numCourses;
        int m = prerequisites.size();

        std::vector<int> head(n + 1, 0);
        std::vector<int> next(m + 1, 0);
        std::vector<int> to(m + 1, 0);
        int cnt = 1;

        for (const auto & e : prerequisites)
        {
            int s = e[1] + 1;
            int t = e[0] + 1;
            next[cnt] = head[s];
            to[cnt] = t;
            head[s] = cnt++;
        }

        std::vector<int> inDegree(n + 1, 0);
        
        for (int s = 1; s <= n; ++s)
        {
            for (int e = head[s]; 0 < e; e = next[e])
            {
                ++inDegree[to[e]];
            }
        }

        std::vector<int> ans;
        ans.reserve(n);

        std::priority_queue<int, std::vector<int>, std::greater<int>> heap;

        for (int i = 1; i <= n; ++i)
        {
            if (!inDegree[i])
            {
                heap.push(i);
            }
        }

        while (!heap.empty())
        {
            int curr = heap.top();
            heap.pop();
            ans.emplace_back(curr - 1);

            for (int e = head[curr]; 0 < e; e = next[e])
            {
                if (!--inDegree[to[e]])
                {
                    heap.push(to[e]);
                }
            }
        }

        return ans.size() == n ? ans : std::vector<int>();
    }
};