class Solution 
{
public:
    int visiblePoints(vector<vector<int>> & points, int angle, vector<int> & location) 
    {
        vector<double> theta;
        theta.reserve(points.size() << 1);

        int numOrigin = 0;

        for (const auto & p : points)
        {
            int x = p[0] - location[0];
            int y = p[1] - location[1];

            if (x == 0 and y == 0)
            {
                ++numOrigin;
                continue;
            }

            double degree = atan2(y, x) * M_1_PIl * 180.0;
            
            if (degree < 0.0)
            {
                degree += 360.0;
            }

            theta.emplace_back(degree);
        }

        sort(theta.begin(), theta.end());

        int oldSize = theta.size();
        theta.insert(theta.end(), theta.cbegin(), theta.cend());

        for (int i = oldSize; i != theta.size(); ++i)
        {
            theta[i] += 360.0;
        }

        // for (auto f : theta) cout << f << ' ';
        // cout << '\n';
        // cout << "windowSize = " << angle << '\n';

        int ans = 0;

        for (int i = 0, j = 0; i != theta.size(); ++i)
        {
            while (j < theta.size() and theta[j] - theta[i] <= angle + kEps)
            {
                ++j;
            }

            ans = max(ans, j - i);
        }

        return ans + numOrigin;
    }

private:
    static constexpr double kEps = 1e-9;
};