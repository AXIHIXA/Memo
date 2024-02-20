class Solution
{
public:
    int mostBooked(int n, std::vector<std::vector<int>> & meetings)
    {
        std::vector<int> meetingsHeld(n, 0);
        
        std::priority_queue<int, std::vector<int>, std::greater<int>> freeRooms;
        for (int i = 0; i < n; ++i) freeRooms.emplace(i);

        // 0 <= start_i. 
        // {End time of current meeting, Meeting room ID}
        std::priority_queue<
                std::pair<long long, int>, 
                std::vector<std::pair<long long, int>>, 
                std::greater<std::pair<long long, int>> 
        > usedRooms;

        std::sort(meetings.begin(), meetings.end(), [](const auto & a, const auto & b)
        {
            return a[0] < b[0];
        });

        for (const auto & meeting : meetings)
        {
            while (!usedRooms.empty() && usedRooms.top().first <= meeting[0])
            {
                int room = usedRooms.top().second;
                usedRooms.pop();
                freeRooms.emplace(room);
            }
            
            if (!freeRooms.empty())
            {
                int room = freeRooms.top();
                freeRooms.pop();
                ++meetingsHeld[room];
                usedRooms.emplace(meeting[1], room);
            }
            else
            {
                auto [freeTime, room] = usedRooms.top();
                usedRooms.pop();
                ++meetingsHeld[room];
                usedRooms.emplace(freeTime + meeting[1] - meeting[0], room);
            }
        }

        return std::max_element(meetingsHeld.cbegin(), meetingsHeld.cend()) - meetingsHeld.cbegin();
    }
};