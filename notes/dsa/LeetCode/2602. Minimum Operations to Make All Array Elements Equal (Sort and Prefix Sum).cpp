static const int init = []
{
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);
    std::cout.tie(nullptr);
    std::setvbuf(stdin, nullptr, _IOFBF, 1 << 20);
    std::setvbuf(stdout, nullptr, _IOFBF, 1 << 20);
    return 0;
}();

class Solution
{
public:
    std::vector<long long> minOperations(std::vector<int> & nums, std::vector<int> & queries)
    {
        auto n = static_cast<const int>(nums.size());
        auto m = static_cast<const int>(queries.size());

        std::sort(nums.begin(), nums.end());
        std::vector<long long> ps(n + 1, 0);
        std::inclusive_scan(nums.cbegin(), nums.cend(), ps.begin() + 1, std::plus<>(), 0LL);

        std::vector<long long> ans(m);

        for (int q = 0; q < m; ++q)
        {
            long long t = queries[q];
            long long k = std::lower_bound(nums.cbegin(), nums.cend(), t) - nums.cbegin();
            ans[q] = k * t - ps[k] + ps[n] - ps[k]- (n - k) * t;
        }

        return ans;
    }
};