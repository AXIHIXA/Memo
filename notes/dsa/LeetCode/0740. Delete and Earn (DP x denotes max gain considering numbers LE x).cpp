class Solution
{
public:
    int deleteAndEarn(std::vector<int> & nums)
    {
        // gain[x]: Points earned by removing number x
        std::unordered_map<int, int> gain;
        int maximum = 0;

        for (int x : nums)
        {
            gain[x] += x;
            maximum = std::max(maximum, x);
        }

        auto gtd = [&gain](int x)
        {
            auto it = gain.find(x);
            return it == gain.end() ? 0 : it->second;
        };

        // dp[x]: Max gain considering numbers <= x. 
        std::vector<int> dp(maximum + 1, 0);
        dp[1] = gtd(1);

        for (int x = 2; x < maximum + 1; ++x)
        {
            dp[x] = std::max(dp[x - 1], dp[x - 2] + gtd(x));
        }

        return dp[maximum];
    }
};