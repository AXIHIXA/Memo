class Solution
{
public:
    int findMaximizedCapital(int k, int w, std::vector<int> & profits, std::vector<int> & capital)
    {
        auto n = static_cast<int>(profits.size());

        std::vector<std::pair<int, int>> projects;
        projects.reserve(n);

        for (int i = 0; i < n; ++i)
        {
            projects.emplace_back(capital[i], profits[i]);
        }

        std::sort(projects.begin(), projects.end());

        int ptr = 0;
        std::priority_queue<int> heap;
        
        for (int i = 0; i < k; ++i)
        {
            for (; ptr < n && projects[ptr].first <= w; ++ptr)
            {
                heap.emplace(projects[ptr].second);
            }

            if (heap.empty()) break;
            w += heap.top();
            heap.pop();
        }

        return w;
    }
};