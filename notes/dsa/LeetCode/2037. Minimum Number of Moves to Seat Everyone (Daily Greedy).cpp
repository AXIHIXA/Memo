class Solution 
{
public:
    int minMovesToSeat(std::vector<int> & seats, std::vector<int> & students)
    {
        auto n = static_cast<const int>(seats.size());
        
        std::sort(seats.begin(), seats.end());
        std::sort(students.begin(), students.end());

        int ans = 0;

        for (int i = 0; i < n; ++i)
        {
            ans += std::abs(seats[i] - students[i]);
        }

        return ans;
    }
};