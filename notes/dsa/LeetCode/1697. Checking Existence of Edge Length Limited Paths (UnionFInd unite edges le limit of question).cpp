class Solution
{
public:
    std::vector<bool> distanceLimitedPathsExist(
            int n, 
            std::vector<std::vector<int>> & edgeList, 
            std::vector<std::vector<int>> & queries) 
    {
        auto m = static_cast<const int>(edgeList.size());
        auto k = static_cast<const int>(queries.size());
        
        auto cmp = [](const auto & a, const auto & b) -> bool
        {
            return a[2] < b[2];
        };

        std::sort(edgeList.begin(), edgeList.end(), cmp);
        questions.reserve(k);

        for (int i = 0; i < k; ++i)
        {
            questions.push_back({queries[i][0], queries[i][1], queries[i][2], i});
        }

        std::sort(questions.begin(), questions.end(), cmp);
        root.resize(n);
        std::iota(root.begin(), root.end(), 0);

        std::vector<bool> ans(k);

        for (int i = 0, j = 0; i < k; ++i)
        {
			// i : 问题编号
			// j : 边的编号
			for ( ; j < m && edgeList[j][2] < questions[i][2]; ++j)
            {
				unite(edgeList[j][0], edgeList[j][1]);
			}

			ans[questions[i][3]] = connected(questions[i][0], questions[i][1]);
		}

        return ans;
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
        root[rx] = ry;
    }

    bool connected(int x, int y)
    {
        return find(x) == find(y);
    }

    std::vector<std::array<int, 4>> questions;

    std::vector<int> root;
};