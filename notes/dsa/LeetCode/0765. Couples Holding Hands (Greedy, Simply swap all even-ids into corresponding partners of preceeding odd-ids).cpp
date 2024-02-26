class Solution
{
public:
    int minSwapsCouples(std::vector<int> & row)
    {
        auto m = static_cast<const int>(row.size());
        
        int ans = 0;

        // Greedy: Simply swap even-ids with the corresponding partner of the preceeding odd-id. 
        for (int i = 0; i < m; i += 2)
        {
            int x = row[i];
            int y = x ^ 1;  // x's partner is x ^ 1. 

            if (row[i + 1] == y)
            {
                continue;
            }

            ++ans;

            for (int j = i + 1; j < m; ++j)
            {
                if (row[j] == y)
                {
                    std::swap(row[j], row[i + 1]);
                }
            }
        }
        
        return ans;
    }
};