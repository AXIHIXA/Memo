class Solution
{
public:
    int findRadius(std::vector<int> & houses, std::vector<int> & heaters)
    {
        auto n = static_cast<const int>(houses.size());
        auto m = static_cast<const int>(heaters.size());

        std::sort(houses.begin(), houses.end());
        std::sort(heaters.begin(), heaters.end());

        int ans = 0;

        for (int i = 0, j = 0; i < n; ++i)
        {
            while (j < m - 1 && heaters[j] < houses[i])
            {
                ++j;
            }
            
            int left = 0 < j ? std::abs(heaters[j - 1] - houses[i]) : kIntMax;
            int right = std::abs(heaters[j] - houses[i]);
            ans = std::max(ans, std::min(left, right));
        }
        
        return ans;
    }

private:
    static constexpr int kIntMax = std::numeric_limits<int>::max();
};