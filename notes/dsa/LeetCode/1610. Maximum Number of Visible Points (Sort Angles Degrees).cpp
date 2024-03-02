class Solution 
{
public:
    int visiblePoints(
            std::vector<std::vector<int>> & points, 
            int angle, 
            std::vector<int> & location) 
    {
        auto n = static_cast<const int>(points.size());
        std::vector<double> pts;
        pts.reserve(n << 1);

        // Location may be one of the points!
        int onViewPos = 0;

        double x0 = location[0];
        double y0 = location[1];

        for (int i = 0; i < n; ++i)
        {
            double x1 = points[i][0];
            double y1 = points[i][1];

            if (points[i] == location)
            {
                ++onViewPos;
            }
            else
            {
                pts.emplace_back(std::atan2(y1 - y0, x1 - x0) * 180.0 * kOneOverPi);
            }
        }
        
        // Make sure we don't count one point twice when cycling!
        std::sort(pts.begin(), pts.end());
        auto m = static_cast<const int>(pts.size());

        std::transform(pts.cbegin(), pts.cend(), std::back_inserter(pts), [](double p)
        {
            return p + 360.0;
        });

        int ans = 0;

        for (int i = 0, rr = 0; i < n; ++i)
        {
            while (rr < i + m && pts[rr] - pts[i] < angle + kEps)
            {
                ++rr;
            }

            ans = std::max(ans, rr - i);
        }

        return ans + onViewPos;
    }

private:
    static constexpr double kEps = 1e-9;

    static constexpr double kOneOverPi = M_1_PIf64;
};