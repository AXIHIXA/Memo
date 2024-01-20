class Solution 
{
public:
    int lengthOfLongestSubstring(std::string s) 
    {
        int n = s.size();
        if (n == 0) return 0;

        // For the current longest substr ending at s[j], and considering s[j + 1], 
        // If s[j + 1] occurs more than once, 
        // drop the leftmost characters until there are no duplicate characters.

        // lmi[c] denotes the index of the leftmost char c so far. 
        std::vector<int> lmi(10 + std::numeric_limits<char>::max(), -1);
        lmi[s[0]] = 0;

        int ans = 1;

        for (int ll = 0, rr = 0; rr < n; ++rr)
        {
            if (int ic = lmi[s[rr]]; ll <= ic) ll = std::min(ic + 1, rr);
            ans = std::max(ans, rr - ll + 1);
            lmi[s[rr]] = rr;
        }

        return ans;
    }
};