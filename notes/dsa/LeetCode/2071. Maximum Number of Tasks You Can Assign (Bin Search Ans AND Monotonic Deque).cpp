class Solution
{
public:
    int maxTaskAssign(std::vector<int> & tasks, std::vector<int> & workers, int pills, int strength)
    {
        auto n = static_cast<const int>(tasks.size());
        auto m = static_cast<const int>(workers.size());
        
        std::sort(tasks.begin(), tasks.end());
        std::sort(workers.begin(), workers.end());

        // Whether we could finish x tasks. 
        // Try finishing top-x easiest tasks with top-x skillful workers. 
        // Assign each worker with one task, one worker by one. 
        // Could finish iff. number of pills needed <= pills. 
        auto doable = [&tasks, &workers, m, pills, strength](int x) -> bool
        {
            static constexpr int kDeqSize = 50'010;
            static std::array<int, kDeqSize> deq;
            int dl = 0, dr = 0;

            int pillsNeeded = 0;

            for (int w = m - x, t = 0; w < m; ++w)
            {
                // Unlock tasks without taking pills.
                for ( ; t < x && tasks[t] <= workers[w]; ++t)
                {
                    deq[dr++] = t;
                }

                if (dl < dr && tasks[deq[dl]] <= workers[w])
                {
                    ++dl;
                }
                else
                {
                    // Take pills.
                    for (; t < x && tasks[t] <= workers[w] + strength; ++t)
                    {
                        deq[dr++] = t;
                    }

                    if (dl < dr) 
                    {
                        ++pillsNeeded;
                        --dr;
                    } 
                    else 
                    {
                        return false;
                    }
                }
            }

            return pillsNeeded <= pills;
        };

        // Bin search. 
        int ans = 0;

        for (int lo = 0, hi = std::min(n, m) + 1; lo < hi; )
        {
            int mi = lo + ((hi - lo) >> 1);

            if (doable(mi))
            {
                ans = std::max(ans, mi);
                lo = mi + 1;
            }
            else
            {
                hi = mi;
            }
        }

        return ans;
    }
};