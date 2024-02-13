class Solution
{
public:
    int minimumTotal(std::vector<std::vector<int>> & triangle)
    {
        auto rows = static_cast<int>(triangle.size());

        std::vector<int> dp(rows, kBigInt);
        dp[0] = triangle[0][0];

        std::vector<int> temp(rows);

        for (int i = 1; i < rows; ++i)
        {
            for (int j = 0; j <= i; ++j)
            {
                int topLeft = j == 0 ? kBigInt : dp[j - 1];
                temp[j] = std::min(topLeft, dp[j]) + triangle[i][j];
            }

            std::copy(temp.data(), temp.data() + i + 1, dp.data());
        }

        return *std::min_element(dp.cbegin(), dp.cend());
    }

private:
    static constexpr int kBigInt = 20000;
};