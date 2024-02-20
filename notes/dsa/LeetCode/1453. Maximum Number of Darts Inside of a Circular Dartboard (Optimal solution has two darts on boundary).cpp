class Solution
{
public:
    static constexpr double eps = 1e-5;

    struct Point2d
    {
        Point2d(double x, double y) : x(x), y(y) {}
        double x = 0.0;
        double y = 0.0;
    };

    int numPoints(std::vector<std::vector<int>> & darts, int r)
    {
        // So there must be one optimal solution
        // that has at least two darts on the circle boundary. 
        auto n = static_cast<const int>(darts.size());
        double R = r;

        std::vector<Point2d> points;
        points.reserve(n);

        for (const auto & dart : darts)
        {
            points.emplace_back(dart[0], dart[1]);
        }

        int ans = 1;

        for (int i = 0; i < n; ++i)
        {
            for (int j = i + 1; j < n; ++j)
            {
                if (2.0 * R + eps < distance(points[i], points[j]))
                {
                    continue;
                }

                auto [c1, c2] = centers(points[i], points[j], R);

                int cnt1 = 0;
                int cnt2 = 0;

                for (int k = 0; k < n; ++k)
                {
                    if (distance(points[k], c1) <= R + eps)
                    {
                        ++cnt1;
                    }

                    if (distance(points[k], c2) <= R + eps)
                    {
                        ++cnt2;
                    }
                }

                ans = std::max({ans, cnt1, cnt2});
            }
        }

        return ans;
    }

    double distance(const Point2d & a, const Point2d & b)
    {
        return std::sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
    }

    std::pair<Point2d, Point2d> centers(const Point2d & a, const Point2d & b, double r)
    {
        Point2d m(0.5 * (a.x + b.x), 0.5 * (a.y + b.y));

        double theta = std::atan2(a.y - b.y, b.x - a.x);
        double tmp = 0.5 * distance(a, b);
        double d = std::sqrt(r * r - tmp * tmp);

        Point2d c1(m.x - d * std::sin(theta), m.y - d * std::cos(theta));
        Point2d c2(m.x + d * std::sin(theta), m.y + d * std::cos(theta));

        return {c1, c2};
    }
};