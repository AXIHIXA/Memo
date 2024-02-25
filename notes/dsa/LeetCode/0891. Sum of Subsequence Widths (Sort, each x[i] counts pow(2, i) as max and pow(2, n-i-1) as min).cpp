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
    int sumSubseqWidths(std::vector<int> & nums)
    {
        auto n = static_cast<const int>(nums.size());
        std::sort(nums.begin(), nums.end());

        // p2[k] = 2**k, modulo 1e9 + 7. 
        std::vector<long long> p2(n);
        p2[0] = 1;
        for (int i = 1; i < n; ++i) p2[i] = (2LL * p2[i - 1]) % p;

        long long ans = 0LL;

        // For each nums[i] (with nums sorted), 
        // it appears as maximum in 2**i subsequences, 
        // and as minimum in 2**(n - i - 1) subsequences. 
        for (int i = n - 1; 0 <= i; --i)
        {
            ans = (ans + (nums[i] * p2[i]) % p) % p;
            ans = (ans - (nums[i] * p2[n - i - 1]) % p + p) % p;
        }

        return ans;
    }

private:
    static constexpr long long p = 1'000'000'007LL;
};