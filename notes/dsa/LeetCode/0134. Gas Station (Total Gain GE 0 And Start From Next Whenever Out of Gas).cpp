class Solution 
{
public:
    int canCompleteCircuit(std::vector<int> & gas, std::vector<int> & cost) 
    {
        auto n = static_cast<const int>(gas.size());
        
        int curGain = 0;
        int totGain = 0;
        int ans = 0;

        for (int i = 0; i < n; ++i)
        {
            curGain += gas[i] - cost[i];
            totGain += gas[i] - cost[i];

            if (curGain < 0)
            {
                ans = i + 1;
                curGain = 0;
            }
        }

        return 0 <= totGain ? ans : -1;
    }
};
