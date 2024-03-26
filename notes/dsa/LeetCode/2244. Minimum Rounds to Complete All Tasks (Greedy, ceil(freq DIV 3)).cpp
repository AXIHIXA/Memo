class Solution
{
public:
    Solution()
    {
        static const int _ = init();
    }

    int minimumRounds(std::vector<int> & tasks)
    {
        using Difficulty = int;
        using Frequency = int;
        std::unordered_map<Difficulty, Frequency> count;

        for (int task : tasks)
        {
            ++count[task];
        }

        int ans = 0;

        for (auto [difficulty, frequency] : count)
        {
            // Greedy. 
            // Each difficulty group is independent. 
            // Optimal plan could has 3k + 1, 3k + 2 or 3k 2-batches. 
            // All satisfies ceil(freq / 3.0). 
            if (frequency == 1)
            {
                return -1;
            }

            ans += (frequency + 2) / 3;
        }

        return ans;
    }

private:    
    static int init()
    {
        std::ios_base::sync_with_stdio(false);
        std::cin.tie(nullptr);
        std::cout.tie(nullptr);
        return 0;
    }
};