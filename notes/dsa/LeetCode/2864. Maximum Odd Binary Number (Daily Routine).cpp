class Solution
{
public:
    std::string maximumOddBinaryNumber(std::string s)
    {
        auto n = static_cast<const int>(s.size());
        std::string ans(s.size(), '0');
        ans.back() = '1';

        int i = 0;
        while (s[i] != '1') ++i;
        ++i;
        for (int j = 0; i < n; ++i) if (s[i] == '1') ans[j++] = '1';

        return ans;
    }
};