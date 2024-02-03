class Solution
{
public:
    int lengthOfLIS(std::vector<int> & nums)
    {
        // Intelligently build a subsequence, 
        // with binary search optimization. 
        int n = nums.size();
        if (n == 1) return 1;

        // Does not guarantee a valid subsequence, 
        // but the length is always correct!
        std::vector<int> seq {nums[0]};
        seq.reserve(n);

        for (int i = 1; i < n; ++i)
        {
            if (seq.back() < nums[i])
            {
                seq.emplace_back(nums[i]);
            }
            else
            {
                auto it = std::upper_bound(seq.begin(), seq.end(), nums[i], std::less_equal<int>());
                *it = nums[i];
            }
        }

        return seq.size();
    }

private:
    static int lengthOfLISDP(std::vector<int> & nums)
    {
        // Dynamic programming method
        // with binary search on aux data. 
        int n = nums.size();
        if (n == 1) return 1;
        
        // f[i]: Length of LIS ending at f[i]. 
        std::vector<int> f(n, 1);

        // g[i]: Known lowest height of LIS of length i + 1. 
        // g is naturally strictly increasing. 
        std::vector<int> g(n, std::numeric_limits<int>::max());
        g[0] = nums[0];

        for (int i = 1, gr = 1; i < n; ++i)
        {
            // f[i] = 1 + max f[0] -- f[i - 1] s.t. height < nums[i]. 
            // Find this max f with restriction on height by binary search on g. 

            // Finds 1st element s.t. value comp element == true,  
            // i.e. 1st height in g >= nums[i]. 
            auto it = std::upper_bound(
                    g.cbegin(), 
                    g.cbegin() + gr, 
                    nums[i], 
                    std::less_equal<int>()
            );

            // it - g.cbegin() naturally >= 0. 
            f[i] = 1 + it - g.cbegin();
            g[f[i] - 1] = std::min(g[f[i] - 1], nums[i]);
            gr = std::max(gr, f[i]);
        }

        return *std::max_element(f.cbegin(), f.cend());
    }
};