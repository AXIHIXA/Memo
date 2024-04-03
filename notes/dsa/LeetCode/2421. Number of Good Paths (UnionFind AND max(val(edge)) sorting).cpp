class Solution
{
public:
    int numberOfGoodPaths(std::vector<int> & vals, std::vector<std::vector<int>> & edges)
    {
        std::iota(root.begin(), root.end(), 0);
        std::fill(maxCnt.begin(), maxCnt.end(), 1);
        
        this->vals = vals.data();
        
        std::sort(edges.begin(), edges.end(), [&vals](const auto & a, const auto & b)
        {
            return std::max(vals[a[0]], vals[a[1]]) < std::max(vals[b[0]], vals[b[1]]);
        });

        auto ans = static_cast<int>(vals.size());

        for (const auto & edge : edges)
        {
            ans += unite(edge[0], edge[1]);
        }

        return ans;
    }

private:
    int find(int x)
    {
        if (root[x] == x) return x;
        return root[x] = find(root[x]);
    }

    int unite(int x, int y)
    {
        // fx : x所在集团的代表节点，同时也是x所在集团的最大值下标
        // fy : y所在集团的代表节点，同时也是y所在集团的最大值下标
		int fx = find(x);
		int fy = find(y);

		int path = 0;

		if (vals[fx] > vals[fy]) 
        {
			root[fy] = fx;
		} 
        else if (vals[fx] < vals[fy]) 
        {
			root[fx] = fy;
		} 
        else 
        {
			// 两个集团最大值一样！
			path = maxCnt[fx] * maxCnt[fy];
			root[fy] = fx;
			maxCnt[fx] += maxCnt[fy];
		}

		return path;
    }

private:
    static constexpr int kMaxSize = 30'010;
    static std::array<int, kMaxSize> root;
    static std::array<int, kMaxSize> maxCnt;

private:
    int * vals = nullptr;
};

std::array<int, Solution::kMaxSize> Solution::root = {0};
std::array<int, Solution::kMaxSize> Solution::maxCnt = {0};
