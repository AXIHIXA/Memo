class Solution
{
public:
    std::vector<int> restoreArray(std::vector<std::vector<int>> & adjacentPairs)
    {
        auto n = static_cast<const int>(adjacentPairs.size());
        std::unordered_map<int, std::vector<int>> hashMap;

        for (const auto & ap : adjacentPairs)
        {
            int u = ap[0];
            int v = ap[1];
            hashMap[u].emplace_back(v);
            hashMap[v].emplace_back(u);
        }

        int front = -1;

        for (const auto & [k, v] : hashMap)
        {
            if (v.size() == 1)
            {
                front = k;
                break;
            }
        }

        std::vector<int> ans(n + 1);
        ans[0] = front;

        for (int i = 1; i < n + 1; ++i)
        {
            const auto & vec = hashMap.at(ans[i - 1]);

            if (vec.size() == 1 || vec[1] == ans[i - 2])
            {
                ans[i] = vec[0];
            }
            else
            {
                ans[i] = vec[1];
            }
        }

        return ans;
    }
};
