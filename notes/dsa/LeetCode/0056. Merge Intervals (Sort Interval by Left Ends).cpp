class Solution 
{
public:
    vector<vector<int>> merge(vector<vector<int>> & intervals) 
    {
        std::sort(intervals.begin(), intervals.end(), [](auto & a, auto & b)
        {
            return a[0] < b[0];
        });

        std::vector<std::vector<int>> merged;

        for (const auto interval : intervals)
        {
            if (merged.empty() || merged.back()[1] < interval[0])
            {
                merged.emplace_back(interval);
            }
            else
            {
                if (merged.back()[1] < interval[1])
                {
                    merged.back()[1] = interval[1];
                }
            }
        }

        return merged;
    }
};