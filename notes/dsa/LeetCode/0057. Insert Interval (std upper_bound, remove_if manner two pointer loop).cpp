class Solution
{
public:
    std::vector<std::vector<int>> 
    insert(std::vector<std::vector<int>> & intervals, std::vector<int> & newInterval)
    {
        auto it = std::upper_bound(intervals.begin(), intervals.end(), newInterval);
        it = intervals.insert(it, newInterval);

        // std::printf("[%d,%d]\n", it->front(), it->back());
        // for (const auto & in : intervals)
        //     std::printf("[%d,%d],", in.front(), in.back());
        // std::printf("\n\n");

        // Merge
        if (it != intervals.begin()) --it;

        for (auto jt = it; jt != intervals.end(); ++jt)
        {
            if (jt->front() <= it->back())
            {
                it->back() = std::max(it->back(), jt->back());
            }
            else
            {
                if (it != jt) *(++it) = *jt;
            }

            // for (const auto & in : intervals)
            //     std::printf("[%d,%d],", in.front(), in.back());
            // std::printf("\n");
        }

        if (it != intervals.end()) intervals.erase(++it, intervals.end());

        return intervals;
    }
};