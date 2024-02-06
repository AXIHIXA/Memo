class Solution
{
public:
    std::vector<std::string> letterCombinations(std::string digits)
    {
        int m = digits.size();
        if (m == 0) return {};

        std::vector<std::string> ans;
        std::string lc(m, '\0');

        std::function<void (int)> dfs = [&](int i)
        {
            if (i == digits.size())
            {
                ans.emplace_back(lc);
                return;
            }

            for (char cc = d2c[digits[i] - '0']; cc < d2c[digits[i] - '0' + 1]; ++cc)
            {
                char original = lc[i];
                lc[i] = cc;
                dfs(i + 1);
                lc[i] = original;
            }
        };

        dfs(0);

        return ans;
    }

private:
    static constexpr std::array<char, 11> d2c 
    {'\0', '\0', 'a', 'd', 'g', 'j', 'm', 'p', 't', 'w', 'z' + 1};
};