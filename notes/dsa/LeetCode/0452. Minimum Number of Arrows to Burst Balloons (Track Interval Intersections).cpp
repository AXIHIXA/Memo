class Solution
{
public:
    int findMinArrowShots(std::vector<std::vector<int>> & points)
    {
        int n = points.size();
        if (n == 1) return 1;

        std::sort(points.begin(), points.end());

        int ans = 1;
        int xx = points.front().front(), yy = points.front().back();

        for (int i = 1; i < n; ++i)
        {
            int x = points[i].front(), y = points[i].back();

            if (x <= yy)
            {
                xx = x;
                yy = std::min(y, yy);
            }
            else
            {
                xx = x;
                yy = y;
                ++ans;
            }
        }

        return ans;
    }
};