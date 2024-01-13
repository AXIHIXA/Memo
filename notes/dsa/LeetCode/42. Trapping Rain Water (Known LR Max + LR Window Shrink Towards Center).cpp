class Solution 
{
public:
    int trap(vector<int> & height) 
    {
        auto n = static_cast<int>(height.size());

        if (n == 1)
        {
            return 0;
        }
        
        int ans = 0;
        int ll = 0, rr = n - 1;
        int leftMax = height[0], rightMax = height[n - 1];

        while (ll < rr)
        {
            if (height[ll] < height[rr])
            {
                if (height[ll] < leftMax)
                {
                    ans += leftMax - height[ll];
                }
                else
                {
                    leftMax = height[ll];
                }

                ++ll;
            }
            else
            {
                if (height[rr] < rightMax)
                {
                    ans += rightMax - height[rr];
                }
                else
                {
                    rightMax = height[rr];
                }

                --rr;
            }
        }

        return ans;
    }
};