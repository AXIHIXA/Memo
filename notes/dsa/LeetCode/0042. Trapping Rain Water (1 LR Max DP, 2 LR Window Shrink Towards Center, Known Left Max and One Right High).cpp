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
    int trap(vector<int> & height) 
    {
        auto n = static_cast<int>(height.size());
        if (n == 1) return 0;
        
        int ans = 0;
        int ll = 0, rr = n - 1;
        int leftMax = height[0], rightMax = height[n - 1];

        // One-pass Two-pointer Window Rolling DP. 
        // We know leftMax when rolling from left to right. 
        // As long as there is one bar on the right that is higher than leftMax, 
        // the water trapped in current bar is fixed.
        // We don't have to know rightMax explicitly. 
        while (ll < rr)
        {
            if (height[ll] < height[rr])
            {
                ++ll;
                leftMax = std::max(leftMax, height[ll]);
                ans += leftMax - height[ll];
            }
            else
            {
                --rr;
                rightMax = std::max(rightMax, height[rr]);
                ans += rightMax - height[rr];
            }
        }

        return ans;
    }
};