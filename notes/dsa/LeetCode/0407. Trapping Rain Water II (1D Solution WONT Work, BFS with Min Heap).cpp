class Solution
{
public:
    struct Pos
    {
        friend bool operator >(const Pos & a, const Pos & b) { return a.h > b.h; }

        Pos() = default;
        Pos(int x, int y, int h) : x(x), y(y), h(h) {}

        int x = 0; 
        int y = 0; 
        int h = 0;
    };

    int trapRainWater(std::vector<std::vector<int>> & heightMap)
    {
        // So the 1D solution won't work, consider this counter example: 
        // 4 4 4 4 4 
        // 4 3 3 3 4
        // 4 4 4 3 4
        // h[1][1] has x and y bounds higher but it won't trap any rain water. 

        // Solution: BFS with min heap. 
        // Push all boundary cells to heap, and for each new cell:  
        // (1) If it's lower than min heap top 
        //     (lowest boundary surrounding it), 
        //     it will trap water; 
        // (2) Simply push max(top height, this cell's height) 
        //     into heap as a boundary candidate. 
        
        auto m = static_cast<const int>(heightMap.size());
        auto n = static_cast<const int>(heightMap.front().size());

        std::vector visited(m, std::vector<unsigned char>(n, false));
        std::priority_queue<Pos, std::vector<Pos>, std::greater<Pos>> heap;

        for (int i = 0; i < m; ++i)
        {
            visited[i][0] = true;
            heap.emplace(i, 0, heightMap[i][0]);

            visited[i][n - 1] = true;
            heap.emplace(i, n - 1, heightMap[i][n - 1]);
        }

        for (int j = 1; j < n - 1; ++j)
        {
            visited[0][j] = true;
            heap.emplace(0, j, heightMap[0][j]);

            visited[m - 1][j] = true;
            heap.emplace(m - 1, j, heightMap[m - 1][j]);
        }

        int ans = 0;

        while (!heap.empty())
        {
            auto [x, y, h] = heap.top();
            heap.pop();

            for (int d = 0, x1, y1; d < 4; ++d)
            {
                x1 = x + dx[d];
                y1 = y + dy[d];
                if (x1 < 0 || m <= x1 || y1 < 0 || n <= y1 || visited[x1][y1]) continue;

                visited[x1][y1] = true;
                if (heightMap[x1][y1] < h) ans += h - heightMap[x1][y1];
                heap.emplace(x1, y1, std::max(h, heightMap[x1][y1]));
            }
        }

        return ans;
    }

private:
    static constexpr std::array<int, 4> dx {1, 0, -1, 0};
    static constexpr std::array<int, 4> dy {0, 1, 0, -1};
};