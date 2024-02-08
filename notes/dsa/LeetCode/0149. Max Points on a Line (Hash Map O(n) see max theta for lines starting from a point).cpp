class Solution
{
public:
    int maxPoints(std::vector<std::vector<int>> & points)
    {
        auto n = static_cast<int>(points.size());
        if (n == 1) return 1;
        
        int ans = 2;
        std::unordered_map<double, int> hmp(n * 3);

        for (int i = 0; i < n; ++i)
        {
            hmp.clear();

            for (int j = 0; j < n; ++j)
            {
                if (j == i) continue;
                double theta = std::atan2(points[j][0] - points[i][0], points[j][1] - points[i][1]);
                auto it = hmp.find(theta);
                if (it == hmp.end()) hmp.emplace(theta, 2);
                else ++it->second;
            }

            for (auto [theta, count] : hmp) 
            {
                ans = std::max(ans, count);
            }
        }

        return ans;
    }
};