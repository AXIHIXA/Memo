class Solution
{
public:
    int maxArea(std::vector<int> & h)
    {
        auto n = static_cast<const int>(h.size());
        int ans = 0;

        for (int ll = 0, rr = n - 1; ll < rr; )
        {
            if (h[ll] < h[rr])
            {
                ans = std::max(ans, h[ll] * (rr - ll));
                ++ll;
            }
            else
            {
                ans = std::max(ans, h[rr] * (rr - ll));
                --rr;
            }
        }

        return ans; 
    }
};