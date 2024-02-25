class Solution 
{
public:
    int canCompleteCircuit(std::vector<int> & gas, std::vector<int> & cost) 
    {
        int curGain = 0, totGain = 0, ans = 0;

        for (int i = 0, n = gas.size(); i != n; ++i)
        {
            int gain = gas[i] - cost[i];
            curGain += gain;
            totGain += gain;

            if (curGain < 0)
            {
                ans = i + 1;
                curGain = 0;
            }
        }

        return 0 <= totGain ? ans : -1;
    }
};
