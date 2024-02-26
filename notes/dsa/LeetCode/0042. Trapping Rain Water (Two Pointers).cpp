class Solution
{
public:
    int trap(std::vector<int> & height)
    {
        auto n = static_cast<const int>(height.size());
        int ans = 0;

        // Water trappable at the current cell is determined by min of leftMax and rightMax. 
        for (int ll = 0, rr = n - 1, hl = height[0], hr = height[n - 1]; ll < rr; )
        {
            if (height[ll] < height[rr])
            {
                ++ll;
                hl = std::max(hl, height[ll]);
                ans += hl - height[ll];
            }
            else
            {
                --rr;
                hr = std::max(hr, height[rr]);
                ans += hr - height[rr];
            }
        }

        return ans;
    }
};