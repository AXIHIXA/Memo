class Solution 
{
public:
    string longestCommonPrefix(vector<string> & strs) 
    {
        if (strs.empty()) return "";

        int minLen = 210;

        for (const auto & s : strs)
        {
            minLen = std::min(minLen, static_cast<int>(s.size()));
        }
        
        for (int i = 0; i < minLen; ++i)
        {
            for (const auto & s : strs) 
            {
                if (s[i] != strs.front()[i])
                {
                    return strs.front().substr(0, i);
                }
            }
        }

        return strs.front().substr(0, minLen);
    }

private:
    string longestCommonPrefixBinSearch(vector<string> & strs) 
    {
        if (strs.empty()) return "";

        int minLen = 210;
        for (const auto & s : strs) minLen = std::min(minLen, static_cast<int>(s.size()));

        int lo = 1, hi = minLen;

        while (lo <= hi)
        {
            int mi = (lo + hi) / 2;

            if (isCommonPrefix(mi, strs)) lo = mi + 1;
            else                          hi = mi - 1;
        }

        return strs.front().substr(0, (lo + hi) / 2);
    }

    static bool isCommonPrefix(int len, const std::vector<std::string> & strs)
    {
        for (int i = 0; i < len; ++i)
        {
            for (const auto & s : strs) 
            {
                if (s[i] != strs.front()[i])
                {
                    return false;
                }
            }   
        }

        return true;
    }
};