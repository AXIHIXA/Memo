class Solution
{
public:
    int fieldOfGreatestBlessing(std::vector<std::vector<int>> & forceField)
    {
        auto n = static_cast<const int>(forceField.size());
        const int m = n << 1;

        // Discretize x and y coordinates.
        std::vector<long long> xs;
        std::vector<long long> ys;
        xs.reserve(m);
        ys.reserve(m);

        for (const auto & ff : forceField)
        {
            long long x = ff[0] << 1;
            long long y = ff[1] << 1;
            long long r = ff[2];

            xs.emplace_back(x - r);
            xs.emplace_back(x + r);
            ys.emplace_back(y - r);
            ys.emplace_back(y + r);
        }

        // Sort, unique, and lower_bound faster than unordered_map. 
        std::sort(xs.begin(), xs.end());
        std::sort(ys.begin(), ys.end());
        xs.erase(std::unique(xs.begin(), xs.end()), xs.end());
        ys.erase(std::unique(ys.begin(), ys.end()), ys.end());

        auto xSpan = static_cast<const int>(xs.size());
        auto ySpan = static_cast<const int>(ys.size());

        // 2D Difference, 1-indexed.
        std::vector diff(xSpan + 2, std::vector<int>(ySpan + 2, 0));

        // 1-indexed.
        auto add = [&diff](int a, int b, int c, int d, int k = 1) mutable
        {
            diff[a][b] += k;
            diff[c + 1][b] -= k;
            diff[a][d + 1] -= k;
            diff[c + 1][d + 1] += k;
        };

        for (const auto & ff : forceField)
        {
            long long x = ff[0] << 1;
            long long y = ff[1] << 1;
            long long r = ff[2];

            long long a = std::lower_bound(xs.cbegin(), xs.cend(), x - r) - xs.cbegin();
            long long b = std::lower_bound(ys.cbegin(), ys.cend(), y - r) - ys.cbegin();
            long long c = std::lower_bound(xs.cbegin(), xs.cend(), x + r) - xs.cbegin();
            long long d = std::lower_bound(ys.cbegin(), ys.cend(), y + r) - ys.cbegin();
            
            // 1-indexed. 
            add(a + 1, b + 1, c + 1, d + 1);
        }
        
        int ans = 0;

        for (int x = 1; x <= xSpan; ++x)
        {
            for (int y = 1; y <= ySpan; ++y)
            {
                diff[x][y] += diff[x - 1][y] + diff[x][y - 1] - diff[x - 1][y - 1];
                ans = std::max(ans, diff[x][y]);
            }
        }

        return ans;
    }
};
