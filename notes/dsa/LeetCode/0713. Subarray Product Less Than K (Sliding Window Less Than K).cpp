class Solution 
{
public:
    Solution()
    {
        static const int _ = init();
    }

    int numSubarrayProductLessThanK(std::vector<int> & nums, int k) 
    {
        auto n = static_cast<const int>(nums.size());

        if (k == 0)
        {
            return 0;
        }

        int ans = 0;

        for (int ll = 0, rr = 0, prod = 1; rr < n; ++rr)
        {
            prod *= nums[rr];
            
            while (ll <= rr && k <= prod)
            {
                prod /= nums[ll++];
            }
            
            // for (int i = ll; i <= rr; ++i)
            // {
            //     std::cout << nums[i] << ' ';
            // }

            // std::cout << '\n';

            // Number of subarrays that end at rr and start at 
            // any element between rr and ll, inclusive.
            ans += rr - ll + 1;
        }

        return ans;
    }

private:
    static int init()
    {
        std::ios_base::sync_with_stdio(false);
        std::cin.tie(nullptr);
        std::cout.tie(nullptr);
        std::setvbuf(stdin, nullptr, _IOFBF, 1 << 20);
        std::setvbuf(stdout, nullptr, _IOFBF, 1 << 20);
        return 0;
    }
};