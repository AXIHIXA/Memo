class Solution
{
public:
    std::vector<std::string> findRelativeRanks(std::vector<int> & score)
    {
        auto n = static_cast<const int>(score.size());
        std::vector<int> rank(n);
        std::iota(rank.begin(), rank.end(), 0);
        std::sort(rank.begin(), rank.end(), [&score](int a, int b)
        {
            return score[a] > score[b];
        });

        std::vector<std::string> ans(n);

        for (int i = 0; i < n; ++i)
        {
            if (i == 0)
            {
                ans[rank[i]] = "Gold Medal";
            }
            else if (i == 1)
            {
                ans[rank[i]] = "Silver Medal";
            }
            else if (i == 2)
            {
                ans[rank[i]] = "Bronze Medal";
            }
            else
            {
                ans[rank[i]] = std::to_string(i + 1);
            }
        }

        return ans;
    }
};