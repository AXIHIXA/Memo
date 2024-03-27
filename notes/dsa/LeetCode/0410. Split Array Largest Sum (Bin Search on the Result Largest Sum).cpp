class Solution
{
public:
    int splitArray(std::vector<int> & nums, int k)
    {
        auto n = static_cast<const int>(nums.size());
        long long sum = std::reduce(nums.cbegin(), nums.cend(), 0LL);
        int ans = 0;

        // 必须让数组arr每一部分的累加和 <= limit，请问划分成几个部分才够!
        auto f = [&nums](long long limit) -> int
        {
            int ans = 1;
            long long sum = 0LL;

            for (int x : nums)
            {
                if (limit < x)
                {
                    return std::numeric_limits<int>::max();
                }

                if (limit < sum + x)
                {
                    ++ans;
                    sum = x;
                }
                else
                {
                    sum += x;
                }
            }

            return ans;
        };

        for (long long lo = 0LL, hi = sum + 1LL; lo < hi; )
        {
            long long mi = lo + ((hi - lo) >> 1LL);
            int need = f(mi);

            if (need <= k)
            {
                ans = mi;
                hi = mi;
            }
            else
            {
                lo = mi + 1;
            }
        }

        return ans;
    }
};