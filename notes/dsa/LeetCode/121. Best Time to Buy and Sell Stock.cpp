class Solution 
{
public:
    int maxProfit(vector<int> & prices) 
    {
        int minPrice = std::numeric_limits<int>::max();
        int maxProfit = 0;
        
        for (int p : prices)
        {
            minPrice = std::min(minPrice, p);
            maxProfit = std::max(maxProfit, p - minPrice);
        }

        return maxProfit;
    }
};