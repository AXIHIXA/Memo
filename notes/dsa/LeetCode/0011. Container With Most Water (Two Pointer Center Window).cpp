int init = []
{
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);
    std::cout.tie(nullptr);
    return 0;
}();

class Solution 
{
public:
    int maxArea(vector<int> & height) 
    {
        int ans = 0;

        for (int ll = 0, rr = height.size() - 1; ll < rr; )
        {
            int area = std::min(height[ll], height[rr]) * (rr - ll);
            ans = std::max(ans, area);
            height[ll] <= height[rr] ? ++ll : --rr;
        }

        return ans;
    }
};