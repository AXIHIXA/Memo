class Solution
{
public:
    int intersectionSizeTwo(std::vector<std::vector<int>> & intervals)
    {
        auto n = static_cast<const int>(intervals.size());
        if (n == 1) return 2;
        
        // rr ascending, ll descending. 
        std::sort(intervals.begin(), intervals.end(), [](const auto & a, const auto & b)
        {
            return a[1] == b[1] ? a[0] > b[0] : a[1] < b[1];
        });

        int ans = 0;
        
        // [ll, rr]: 
        // The rightmost 2 points that may cover the next interval. 
        for (int i = 0, ll = -1, rr = -1; i < n; ++i)
        {
            // Covered by the 2 previous points already. 
            if (intervals[i][0] <= ll) continue;

            if (rr < intervals[i][0])
            {
                // Uncoverable by the 2 previous points. 
                // Add two points at once. 
                ans += 2;
                rr = intervals[i][1];
                ll = rr - 1;
            }
            else
            {
                // Uncoverable by ll. Add one point. 
                ++ans;
                ll = rr;
                rr = intervals[i][1];
            }
        }

        return ans;
    }
};