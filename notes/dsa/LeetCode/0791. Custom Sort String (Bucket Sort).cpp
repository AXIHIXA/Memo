class Solution
{
public:
    std::string customSortString(std::string order, std::string s)
    {
        std::fill_n(count.begin(), 26, 0);
        for (char c : s) ++count[c - 'a'];

        std::string ans(s.size(), '\0');
        int idx = 0;
        for (char c : order) while (0 < count[c - 'a']--) ans[idx++] = c;
        for (int i = 0; i < 26; ++i) while (0 < count[i]--) ans[idx++] = i + 'a';

        return ans;
    }

private:
    static std::array<int, 26> count;
};

std::array<int, 26> Solution::count;
