class Solution
{
public:
    int maxChunksToSorted(std::vector<int> & arr)
    {
        auto n = static_cast<const int>(arr.size());
        int ans = 0;

        // Sliding window. 
        // A window is valid iff. its min and max is exactly the left and right indices. 
        for (int ll = 0, rr = 0, mini = 1000, maxi = -1; rr < n; ++rr)
        {
            mini = std::min(mini, arr[rr]);
            maxi = std::max(maxi, arr[rr]);
            
            if (ll == mini && rr == maxi)
            {
                ++ans;
                ll = rr + 1;
                mini = 1000;
                maxi = -1;
            }
        }

        return ans;
    }
};