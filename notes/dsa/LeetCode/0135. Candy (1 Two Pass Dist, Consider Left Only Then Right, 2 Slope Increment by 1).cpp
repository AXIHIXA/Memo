int init = []
{
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);
    std::cout.tie(nullptr);
    return 0;
}();

class Solution 
{
public:
    int candy(vector<int> & ratings)
    {
        int n = ratings.size();
        if (n <= 1) return n;

        int candies = 0, up = 0, down = 0;

        auto sign = [](int a, int b)
        {
            return a < b ? 1 : (a == b ? 0 : -1);
        };

        auto count = [](int a)
        {
            return a * (a + 1) / 2;
        };

        for (int i = 1, prevSlope = 0; i != n; ++i)
        {
            int slope = sign(ratings[i - 1], ratings[i]);

            if ((0 < prevSlope && slope == 0) || (prevSlope < 0 && 0 <= slope))
            {
                candies += count(up) + count(down) + std::max(up, down);
                up = 0;
                down = 0;
            }

            if (0 < slope)       ++up;
            else if (slope == 0) ++candies;
            else                 ++down;

            prevSlope = slope;
        }

        candies += count(up) + count(down) + std::max(up, down) + 1;

        return candies;
    }

private:
    int candyOnSpace(vector<int> & ratings) 
    {
        int n = ratings.size();
        std::vector<int> candies(n, 1);

        // Distribute candies only considering left neighbors. 
        for (int i = 1; i != n; ++i)
        {
            if (ratings[i - 1] < ratings[i])
            {
                candies[i] = candies[i - 1] + 1;
            }
        }

        // Now distribute more candies, taking right neighbors into account. 
        for (int i = n - 2; 0 <= i; --i)
        {
            if (ratings[i + 1] < ratings[i] && candies[i] <= candies[i + 1])
            {
                candies[i] = candies[i + 1] + 1;
            }
        }

        return std::accumulate(candies.cbegin(), candies.cend(), 0);
    }
};
