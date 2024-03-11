class Solution
{
public:
    int characterReplacement(std::string s, int k)
    {
        auto n = static_cast<const int>(s.size());

        int ans = 0;
        std::array<int, 26> frequency = {0};
        
        // maxFreq controlls size of the sliding window. 
        // The sliding window need not be always valid; 
        // it needs to be valid only when it grows in size (when we update ans). 
        for (int ll = 0, rr = 0, maxFreq = 0; rr < n; ++rr)
        {
            maxFreq = std::max(maxFreq, ++frequency[s[rr] - 'A']);
            if (rr - ll + 1 - maxFreq <= k) ans = std::max(ans, rr - ll + 1);
            else --frequency[s[ll++] - 'A'];
        }

        return ans;
    }
};