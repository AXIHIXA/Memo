class Solution
{
public:
    int findMinMoves(std::vector<int> & machines)
    {
        auto n = static_cast<const int>(machines.size());
        int dresses = std::accumulate(machines.begin(), machines.end(), 0);
        if (dresses % n) return -1;
        int target = dresses / n;

        int curr = 0;
        int ans = 0;

        // Greedy. 
        // Split machines into Left and Right parts. 
        // Movements happen when: 
        // (1) Move from left to right / right to left
        //     when one part has more dresses than the other;
        // (2) Move inside the part itself
        //     when inside one part, one machine has too many dresses. 
        // Max of (1) and (2) is the answer. 
        // Proof: https://leetcode.cn/problems/super-washing-machines/solutions/1022639/chao-ji-xi-yi-ji-by-leetcode-solution-yhej/
        for (int dress : machines)
        {
            curr += dress - target;
            ans = std::max(ans, std::max(std::abs(curr), dress - target));
        }

        return ans;
    }
};