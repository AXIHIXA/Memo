class Solution
{
public:
    std::string frequencySort(std::string s)
    {
        std::vector<std::pair<char, int>> frequency(256);
        for (int i = 0; i < 256; ++i) frequency[i] = {i, 0};
        for (char c : s) ++frequency[c].second;

        std::sort(frequency.begin(), frequency.end(), [](const auto & a, const auto & b)
        {
            return a.second > b.second;
        });

        std::string ans;
        ans.reserve(s.size());

        for (auto [c, f] : frequency)
        {
            for (int i = 0; i < f; ++i)
            {
                ans += c;
            }
        }

        return ans;
    }
};