class Solution
{
public:
    int jobScheduling(
            std::vector<int> & startTime, 
            std::vector<int> & endTime, 
            std::vector<int> & profit)
    {
        auto n = static_cast<const int>(startTime.size());
        
        struct Job
        {
            int startTime;
            int endTime;
            int profit;
        };

        std::vector<Job> jobs;
        jobs.reserve(n + 1);
        jobs.emplace_back(0, 0, 0);

        for (int i = 0; i < n; ++i)
        {
            jobs.emplace_back(startTime[i], endTime[i], profit[i]);
        }

        std::sort(jobs.begin(), jobs.end(), [](const auto & a, const auto & b)
        {
            return a.endTime < b.endTime;
        });

        // dp[i]: Max profit taking jobs whose indices <= i (1-indexed). 
        std::vector<int> dp(n + 10, 0);
        
        for (int i = 1; i <= n; ++i)
        {
            // Not taking job[i]. 
            dp[i] = std::max(dp[i - 1], jobs[i].profit);

            // Leftmost job whose endTime > jobs[i].startTime. 
            auto it = std::upper_bound(
                    jobs.cbegin() + 1, 
                    jobs.cbegin() + i + 1, 
                    jobs[i], 
                    [](const auto & a, const auto & b) { return a.startTime < b.endTime; }
            );

            // Take job i, 
            // indices of prev jobs pickable have endTime <= jobs[i].startTime,
            // in range [1...it - 1]. 
            dp[i] = std::max(dp[i], dp[std::prev(it) - jobs.cbegin()] + jobs[i].profit);
        }

        return dp[n];
    }
};