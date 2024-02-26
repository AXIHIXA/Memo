class Solution
{
public:
    // Monotonic stack. 
    int trap(std::vector<int> & height)
    {
        auto n = static_cast<const int>(height.size());

        int ans = 0;
        std::stack<int> st;
        
        for (int i = 0; i < n; ++i)
        {
            while (!st.empty() && height[st.top()] < height[i])
            {
                int cur = st.top();
                st.pop();
                if (st.empty()) break;

                int distance = i - st.top() - 1;
                int boundedHeight = std::min(height[i], height[st.top()]) - height[cur];
                ans += distance * boundedHeight;
            }

            st.emplace(i);
        }

        return ans;
    }

    // Two pointers. 
    int trap2(std::vector<int> & height)
    {
        auto n = static_cast<const int>(height.size());
        if (n < 3) return 0;

        int ans = 0;

        // Water trappable at the current cell is determined by min of leftMax and rightMax. 
        // For index ll, leftMax is accurate; 
        // for index rr, rightMax is accurate. 
        for (int ll = 1, rr = n - 2, leftMax = height[0], rightMax = height[n - 1]; ll <= rr; )
        {
            if (leftMax <= rightMax)
            {
                ans += std::max(0, leftMax - height[ll]);
                leftMax = std::max(leftMax, height[ll]);
                ++ll;
            }
            else
            {
                ans += std::max(0, rightMax - height[rr]);
                rightMax = std::max(rightMax, height[rr]);
                --rr;
            }
        }

        return ans;
    }
};