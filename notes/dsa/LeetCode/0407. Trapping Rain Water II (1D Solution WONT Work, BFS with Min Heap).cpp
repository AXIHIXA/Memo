class Solution
{
public:
    int trapRainWater(std::vector<std::vector<int>> & heightMap)
    {
        auto m = static_cast<const int>(heightMap.size());
        auto n = static_cast<const int>(heightMap.front().size());

        std::vector visited(m, std::vector<std::uint8_t>(n, false));

        std::priority_queue<
                std::tuple<int, int, int>, 
                std::vector<std::tuple<int, int, int>>,
                std::greater<>
        > minHeap;

        for (int i = 0; i < m; ++i)
        {
            minHeap.emplace(heightMap[i][0], i, 0);
            minHeap.emplace(heightMap[i][n - 1], i, n - 1);

            visited[i][0] = true;
            visited[i][n - 1] = true;
        }

        for (int j = 1; j < n - 1; ++j)
        {
            minHeap.emplace(heightMap[0][j], 0, j);
            minHeap.emplace(heightMap[m - 1][j], m - 1, j);

            visited[0][j] = true;
            visited[m - 1][j] = true;
        }

        int ans = 0;

        while (!minHeap.empty())
        {
            auto [h, x0, y0] = minHeap.top();
            minHeap.pop();

            for (int d = 0; d < 4; ++d)
            {
                int x1 = x0 + dx[d];
                int y1 = y0 + dy[d];

                if (x1 < 0 || m <= x1 || y1 < 0 || n <= y1 || visited[x1][y1])
                {
                    continue;
                }

                ans += std::max(0, h - heightMap[x1][y1]);
                minHeap.emplace(std::max(heightMap[x1][y1], h), x1, y1);
                visited[x1][y1] = true;
            }
        }

        return ans;
    }

private:
    static constexpr std::array<int, 4> dx = {1, 0, -1, 0};
    static constexpr std::array<int, 4> dy = {0, 1, 0, -1};
};

class Solution
{
public:
    struct Pos
    {
        friend bool operator >(const Pos & a, const Pos & b)
        { 
            return a.h > b.h;
        }

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
        std::priority_queue<Pos, std::vector<Pos>, std::greater<Pos>> minHeap;

        for (int i = 0; i < m; ++i)
        {
            visited[i][0] = true;
            minHeap.emplace(i, 0, heightMap[i][0]);

            visited[i][n - 1] = true;
            minHeap.emplace(i, n - 1, heightMap[i][n - 1]);
        }

        for (int j = 1; j < n - 1; ++j)
        {
            visited[0][j] = true;
            minHeap.emplace(0, j, heightMap[0][j]);

            visited[m - 1][j] = true;
            minHeap.emplace(m - 1, j, heightMap[m - 1][j]);
        }

        int ans = 0;

        while (!minHeap.empty())
        {
            auto [x0, y0, h0] = minHeap.top();
            minHeap.pop();

            for (int d = 0; d < 4; ++d)
            {
                int x1 = x0 + dx[d];
                int y1 = y0 + dy[d];

                if (x1 < 0 || m <= x1 || y1 < 0 || n <= y1 || visited[x1][y1])
                {
                    continue;
                }

                ans += std::max(0, h0 - heightMap[x1][y1]);
                visited[x1][y1] = true;
                minHeap.emplace(x1, y1, std::max(h0, heightMap[x1][y1]));
            }
        }

        return ans;
    }

private:
    static constexpr std::array<int, 4> dx {1, 0, -1, 0};
    static constexpr std::array<int, 4> dy {0, 1, 0, -1};
};