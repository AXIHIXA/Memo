class Solution
{
public:
    std::vector<int> corpFlightBookings(std::vector<std::vector<int>> & bookings, int n)
    {
        // Differencing. 
        // Ans = prefix sum of difference array. 
        std::vector<int> diff(n + 2, 0);

        for (const auto & booking : bookings)
        {
            int ll = booking[0];
            int rr = booking[1];
            int seats = booking[2];

            diff[ll] += seats;
            diff[rr + 1] -= seats;
        }

        std::vector<int> ans(n);
        std::inclusive_scan(diff.cbegin() + 1, diff.cend() - 1, ans.begin());

        return ans;
    }
};