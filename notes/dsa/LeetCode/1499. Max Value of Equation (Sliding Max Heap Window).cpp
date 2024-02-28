class Solution
{
public:
    int findMaxValueOfEquation(std::vector<std::vector<int>> & points, int k)
    {
        auto n = static_cast<const int>(points.size());

        std::priority_queue<std::pair<int, int>> heap;
        heap.emplace(points[0][1] - points[0][0], points[0][0]);
        int ans = std::numeric_limits<int>::min();

        // Track max res for each point as the right node. 
        // Max candidate is maintained by max heap. 
        // Pop heap elements when left nodes are too far away. 
        for (int i = 1; i < n; ++i)
        {
            while (!heap.empty() && k < points[i][0] - heap.top().second) heap.pop();
            if (!heap.empty()) ans = std::max(ans, points[i][0] + points[i][1] + heap.top().first);
            heap.emplace(points[i][1] - points[i][0], points[i][0]);
        }

        return ans;
    }
};