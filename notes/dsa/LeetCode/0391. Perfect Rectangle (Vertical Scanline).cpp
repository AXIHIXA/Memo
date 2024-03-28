class Solution
{
public:
    bool isRectangleCover(std::vector<std::vector<int>> & rects)
    {
        auto n = static_cast<const int>(rects.size());
        const int m = n << 1;

        enum Side : int
        {
            kLeft = 0,
            kRight = 1
        };

        struct Edge
        {
            int x;
            int y1;
            int y2;
            Side s;
        };

        std::vector<Edge> edges;
        edges.reserve(m);

        for (const auto & r : rects)
        {
            edges.emplace_back(r[0], r[1], r[3], kLeft);
            edges.emplace_back(r[2], r[1], r[3], kRight);
        }

        std::sort(edges.begin(), edges.end(), [](const auto & a, const auto & b)
        {
            return a.x == b.x ? a.y1 < b.y1 : a.x < b.x;
        });

        using Y1 = int;
        using Y2 = int;
        std::array<std::vector<std::pair<Y1, Y2>>, 2> scanline;

        for (int ll = 0, rr = 0; rr < m; ll = rr)
        {
            while (rr < m && edges[ll].x == edges[rr].x)
            {
                ++rr;
            }
            
            scanline[kLeft].clear();
            scanline[kRight].clear();

            for (int i = ll; i < rr; ++i)
            {
                auto [x, y1, y2, side] = edges[i];
                auto & line = scanline[side];
                
                if (line.empty())
                {
                    line.push_back({y1, y2});
                }
                else
                {
                    if (y1 < line.back().second)
                    {
                        // Two edges overlaps. 
                        return false;
                    }
                    else if (y1 == line.back().second)
                    {
                        // Perfect concatenation of two vertical edges. 
                        line.back().second = y2;
                    }
                    else
                    {
                        line.push_back({y1, y2});
                    }
                }
            }

            // Note it's "rr < m", NOT "ll < m"!
            if (0 < ll && rr < m)
            {
                // Middle scanlines.
                if (scanline[kLeft] != scanline[kRight])
                {
                    return false;
                }
            }
            else
            {
                // First or last scanline. 
                if (scanline[kLeft].size() + scanline[kRight].size() != 1)
                {
                    return false;
                }
            }
        }

        return true;
    }
};