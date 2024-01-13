class Solution 
{
public:
    int maxProfit(vector<int> & prices) 
    {
        int res = 0;

        for (int d = 1; d != prices.size(); ++d)
        {
            res += max(0, prices[d] - prices[d - 1]);
        }

        return res;
    }
};