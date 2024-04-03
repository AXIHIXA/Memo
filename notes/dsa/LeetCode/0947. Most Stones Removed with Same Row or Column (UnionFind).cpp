class Solution
{
public:
    int removeStones(std::vector<std::vector<int>> & stones)
    {
        auto n = static_cast<const int>(stones.size());

        sets = n;
        std::iota(root.begin(), root.end(), 0);
        std::fill(rank.begin(), rank.end(), 0);

        std::unordered_map<int, int> rowFirst;
        std::unordered_map<int, int> colFirst;

        for (int i = 0; i < n; ++i)
        {
            int row = stones[i][0];
			int col = stones[i][1];

			if (!rowFirst.contains(row))
            {
				rowFirst.emplace(row, i);
			}
            else 
            {
				unite(i, rowFirst.at(row));
			}

			if (!colFirst.contains(col))
            {
				colFirst.emplace(col, i);
			} 
            else 
            {
				unite(i, colFirst.at(col));
			}
        }

        return n - sets;
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
        --sets;
        
        if (rank[rx] < rank[ry]) root[rx] = ry;
        else if (rank[ry] < rank[rx]) root[ry] = rx;
        else root[ry] = rx, ++rank[rx];
    }

private:
    static constexpr int kMaxIndex = 1'010;

    static int sets;
    static std::array<int, kMaxIndex> root;
    static std::array<int, kMaxIndex> rank;
};

int Solution::sets = 0;
std::array<int, Solution::kMaxIndex> Solution::root = {0};
std::array<int, Solution::kMaxIndex> Solution::rank = {0};