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
    int lengthOfLongestSubstring(std::string s) 
    {
        int n = s.size();
        if (n == 0) return 0;

        // For the current longest substr ending at s[j], and considering s[j + 1], 
        // If s[j + 1] occurs more than once, 
        // drop the leftmost characters until there are no duplicate characters.

        // rmi[c] denotes the index of the rightmost char c so far. 
        std::vector<int> rmi(10 + std::numeric_limits<char>::max(), -1);
        rmi[s[0]] = 0;

        int ans = 1;

        for (int ll = 0, rr = 0; rr < n; ++rr)
        {
            char c = s[rr];
            if (ll <= rmi[c]) ll = std::min(rr, rmi[c] + 1);
            ans = std::max(ans, rr - ll + 1);
            rmi[c] = rr;
        }

        return ans;
    }
};