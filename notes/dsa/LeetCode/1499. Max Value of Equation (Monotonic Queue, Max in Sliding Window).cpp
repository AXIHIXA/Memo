class Solution
{
public:
    int findMaxValueOfEquation(std::vector<std::vector<int>> & points, int k)
    {
        auto n = static_cast<const int>(points.size());

        // Max (y1 - x1 + y2 + x2) s.t. 0 < x2 - x1 <= k. 
        // Find max y1 - x1 in prefix window of (x2, y2) of size <= k + 1. 

        // Two ways to do this:
        // (1) Monotonic deque, maintain max of y-x in predix window [..) of size k. 
        // (2) Max heap with lazy removal. 

        std::vector<int> deq(n);
        int dl = 0, dr = 0;

        int ans = std::numeric_limits<int>::min();

        auto ymx = [&points](int i) -> int
        {
            return points[i][1] - points[i][0];
        };

        auto ypx = [&points](int i) -> int
        {
            return points[i][1] + points[i][0];
        };

        for (int rr = 0; rr < n - 1; ++rr)
        {
            while (dl < dr && ymx(deq[dr - 1]) <= ymx(rr))
            {
                --dr;
            }

            deq[dr++] = rr;

            int ll = points[rr + 1][0] - k;

            while (dl < dr && points[deq[dl]][0] < ll)
            {
                ++dl;
            }

            if (dl < dr)
            {
                ans = std::max(ans, ypx(rr + 1) + ymx(deq[dl]));
            }
        }

        return ans;
    }
};