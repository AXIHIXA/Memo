class Solution
{
public:
    int countHomogenous(std::string s)
    {
        auto n = static_cast<const int>(s.size());
        long long ans = 0LL;

        for (long long ll = 0LL, rr = 0LL; rr < n; )
        {
            while (rr < n && s[ll] == s[rr]) ++rr;
            ans = (ans + (((1LL + rr - ll) * (rr - ll)) >> 1) % p) % p;
            ll = rr;
        }

        return ans;
    }

private:
    static constexpr long long p = 1'000'000'007LL;
};