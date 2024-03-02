class Solution
{
public:
    bool carPooling(std::vector<std::vector<int>> & trips, int capacity)
    {
        std::fill_n(diffNumPassengers.begin(), 1001, 0);

        for (const auto & trip : trips)
        {
            diffNumPassengers[trip[1]] += trip[0];
            diffNumPassengers[trip[2]] -= trip[0];
        }

        int numPassengers = 0;

        for (int diff : diffNumPassengers)
        {
            numPassengers += diff;
            if (capacity < numPassengers) return false;
        }

        return true;
    }

private:
    static std::array<int, 1001> diffNumPassengers;
};

std::array<int, 1001> Solution::diffNumPassengers;