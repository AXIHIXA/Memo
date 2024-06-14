class Solution
{
public:
    std::vector<int> getMaxMatrix(std::vector<std::vector<int>> & matrix)
    {
        auto n = static_cast<const int>(matrix.size());
        auto m = static_cast<const int>(matrix.front().size());
        
        std::vector<int> num(m);

        int r1, c1, r2, c2;
        int maxSum = std::numeric_limits<int>::min();

        // Brute-force O(n**3). 
        // Enumerate r1 and r2 (O(n**2)), 
        // turn into max subarr subproblems (O(n)). 
        for (int i0 = 0; i0 < n; ++i0)
        {
            std::fill_n(num.begin(), m, 0);
            
            for (int i1 = i0; i1 < n; ++i1)
            {
                for (int j = 0; j < m; ++j)
                {
                    num[j] += matrix[i1][j];
                }

                auto [ll, rr, sum] = maxSubarr(num, m);

                if (maxSum < sum)
                {
                    r1 = i0;
                    c1 = ll;
                    r2 = i1;
                    c2 = rr;
                    maxSum = sum;
                }
            }
        }

        return {r1, c1, r2, c2};
    }

private:
    static std::array<int, 3> maxSubarr(const std::vector<int> & num, int m)
    {
        // std::vector<std::array<int, 3>> dp(m);
        // dp[0] = {0, 0, num[0]};
        
        std::array<int, 3> cur = {0, 0, num[0]};
        std::array<int, 3> ans = cur;

        for (int i = 1; i < m; ++i)
        {
            if (0 < cur[2])
            {
                cur = {cur[0], i, cur[2] + num[i]};
            }
            else
            {
                cur = {i, i, num[i]};
            }

            if (ans[2] < cur[2])
            {
                ans = cur;
            }
        }

        return ans;
    }
};
