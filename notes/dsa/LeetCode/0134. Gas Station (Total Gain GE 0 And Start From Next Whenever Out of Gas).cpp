int init = []
{
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);
    std::cout.tie(nullptr);
    return 0;
}();

class Solution 
{
public:
    int canCompleteCircuit(vector<int> & gas, vector<int> & cost) 
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
