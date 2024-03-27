class UnionFind
{
public:
    UnionFind(int n) : root(n), rank(n, 0)
    {
        std::iota(root.begin(), root.end(), 0);
    }

    void unite(int x, int y)
    {
        int rx = find(x), ry = find(y);
        if (rx == ry) return;

        if (rank[rx] < rank[ry]) root[rx] = ry;
        else if (rank[ry] < rank[rx]) root[ry] = rx;
        else root[ry] = rx, ++rank[rx];
    }

    int find(int x)
    {
        if (x == root[x]) return x;
        return root[x] = find(root[x]);
    }

private:
    std::vector<int> root;
    std::vector<int> rank;
};

class Solution
{
public:
    Solution()
    {
        static const int dummy = fastIoInit();
    }

    int shortestBridge(std::vector<std::vector<int>> & grid)
    {
        auto n = static_cast<const int>(grid.size());
        const int n2 = n * n;
        UnionFind uf(n2);

        auto k = [n](int x, int y) -> int
        {
            return x * n + y;
        };

        for (int x = 0; x < n; ++x)
        {
            for (int y = 0; y < n; ++y)
            {
                if (grid[x][y] == 0)
                {
                    continue;
                }

                for (int d = 0; d < 4; ++d)
                {
                    int x1 = x + dx[d];
                    int y1 = y + dy[d];

                    if (x1 < 0 || n <= x1 || y1 < 0 || n <= y1 || grid[x1][y1] == 0)
                    {
                        continue;
                    }

                    uf.unite(k(x, y), k(x1, y1));
                }
            }
        }

        int a = -1, b = -1;
        std::queue<int> q1, q2;
        std::unordered_map<int, int> m1, m2;

        // Init all queues and hash maps for Bidir BFS. 
        for (int x = 0; x < n; ++x)
        {
            for (int y = 0; y < n; ++y)
            {
                if (grid[x][y] == 0)
                {
                    continue;
                }

                int idx = k(x, y);
                int root = uf.find(idx);

                if (a == -1)
                {
                    a = root;
                }
                else if (a != root && b == -1)
                {
                    b = root;
                }

                if (root == a)
                {
                    q1.emplace(idx);
                    m1.emplace(idx, 0);
                }
                else if (root == b)
                {
                    q2.emplace(idx);
                    m2.emplace(idx, 0);
                }
            }
        }

        // Bidir BFS. 
        auto update = [n, &uf, &k](
                std::queue<int> & que, 
                std::unordered_map<int, int> & m1, 
                std::unordered_map<int, int> & m2
        ) -> int
        {
            auto sz = static_cast<const int>(que.size());

            for (int i = 0; i < sz; ++i)
            {
                int idx = que.front();
                que.pop();

                int x = idx / n;
                int y = idx % n;
                int step = m1.at(idx);

                for (int d = 0; d < 4; ++d)
                {
                    int x1 = x + dx[d];
                    int y1 = y + dy[d];
                    int idx1 = k(x1, y1);

                    if (x1 < 0 || n <= x1 || y1 < 0 || n <= y1 || m1.contains(idx1))
                    {
                        continue;
                    }

                    if (m2.contains(idx1))
                    {
                        return step + 1 + m2.at(idx1);
                    }

                    que.emplace(idx1);
                    m1.emplace(idx1, step + 1);
                }
            }

            return -1;
        };

        // Bidir BFS: 
        // Two queues and two hash maps (visited nodes for each queue). 
        // Each step handles the queue of smaller size. 
        // Done when expansion of queue 1 falls into hash map for queue 2. 
        while (!q1.empty() && !q2.empty())
        {
            int t = q1.size() < q2.size() ? update(q1, m1, m2) : update(q2, m2, m1);

            if (t != -1)
            {
                return t - 1;
            }
        }

        return -1;
    }

private:
    static int fastIoInit()
    {
        std::ios_base::sync_with_stdio(false);
        std::cin.tie(nullptr);
        std::cout.tie(nullptr);
        std::setvbuf(stdin, nullptr, _IOFBF, 1 << 20);
        std::setvbuf(stdout, nullptr, _IOFBF, 1 << 20);
        return 0;
    }

private:
    static constexpr std::array<int, 4> dx = {1, 0, -1, 0};
    static constexpr std::array<int, 4> dy = {0, 1, 0, -1};
};