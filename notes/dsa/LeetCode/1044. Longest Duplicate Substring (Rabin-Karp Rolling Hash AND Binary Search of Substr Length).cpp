class RollingHash
{
public:
    // NOTE: 
    // When using as custom hasher in std::unordered_set<Key, Hash>s:
    //     RollingHash hasher;
    //     std::unordered_set<std::string_view, const RollingHash &> hs(10, hasher);
    // 1. Keys MUST be inserted/queried in a rolling order only;
    // 2. typename Hash MUST be a reference, 
    //    otherwise the unordered_set's hasher will be a copy, 
    //    and hasher.reset() wont't work!
    // 3. opeartor() MUST NOT be noexcept, 
    //    or it will hit unordered_set's internal bugs!
    std::size_t operator()(const std::string_view & s) const
    {
        if (shouldReset)
        {
            // New string to hash. 
            pow = 1LL;
            for (int i = 1; i < s.size(); ++i) pow = (pow * a) % p;

            cur = 0LL;
            for (auto c : s) cur = (cur * a + (c - 'a')) % p;
            
            shouldReset = false;
        }
        else
        {
            // Roll forward by one character. 
            cur = (cur - ((front - 'a') * pow) % p + p) % p;
            cur = (cur * a + (s.back() - 'a')) % p;
        }

        front = s.front();

        return cur;
    };

    void reset()
    {
        shouldReset = true;
    }

private:
    // Commonly-seen large primes for hashing: 1926'0817, 9999'9989, 1e9 + 7, ...
    static constexpr long long p = 19260817LL;
    static constexpr long long a = 26LL;
    
    mutable long long shouldReset = true;
    mutable long long cur = 0LL;
    mutable long long pow = 1LL;
    mutable char front = '\0';
};

class Solution
{
public:
    std::string longestDupSubstring(std::string s)
    {
        auto n = static_cast<const int>(s.size());
        
        std::string_view ans;

        RollingHash hasher;
        std::unordered_set<std::string_view, const RollingHash &> hs(10, hasher);

        int lo = 1;
        int hi = n - 1;

        while (lo <= hi)
        {
            std::size_t len = lo + ((hi - lo) >> 1);
            bool found = false;

            for (int i = 0; i < n - len + 1; ++i)
            {
                auto [it, inserted] = hs.insert({s.data() + i, len});

                if (!inserted)
                {
                    found = true;
                    ans = *it;
                    break;
                }
            }

            if (found) lo = len + 1;
            else       hi = len - 1;

            hs.clear();
            hasher.reset();
        }

        return std::string(ans);
    }
};
