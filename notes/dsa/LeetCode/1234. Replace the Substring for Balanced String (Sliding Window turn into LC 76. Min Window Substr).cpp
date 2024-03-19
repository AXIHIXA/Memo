class Solution
{
public:
    int balancedString(std::string s)
    {
        auto n = static_cast<const int>(s.size());
        const int t = n >> 2;

        // Turn into min substr containing all required chars. 
        std::array<int, 5> count;
        for (char c : s) ++count[ci(c)];
        for (char & c : s) if (count[ci(c)] <= t) c = ' ';
        for (int i = 0; i < 4; ++i) count[i] = std::max(0, count[i] - t);
        count[4] = 0;
        int charsNeeded = std::reduce(count.cbegin(), count.cend() - 1, 0);
        if (charsNeeded == 0) return 0;

        int ans = n + 1;

        for (int ll = 0, rr = 0; rr < n; ++rr)
        {
            if (0 < count[ci(s[rr])]--) --charsNeeded;
            
            while (ll <= rr && charsNeeded == 0)
            {
                ans = std::min(ans, rr - ll + 1);
                if (count[ci(s[ll++])]++ == 0) ++charsNeeded;
            }
        }

        return ans;
    }

private:
    static inline int ci(char c)
    {
        switch (c)
        {
            case 'Q': return 0;
            case 'W': return 1;
            case 'E': return 2;
            case 'R': return 3;
            default:  return 4;
        }
    }
};