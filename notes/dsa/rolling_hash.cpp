#include <string>
#include <string_view>

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
