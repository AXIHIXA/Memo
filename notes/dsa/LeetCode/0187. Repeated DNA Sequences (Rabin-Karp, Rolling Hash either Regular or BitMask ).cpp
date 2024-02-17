struct RollingHash
{
    std::size_t operator()(const std::string_view & s) const
    {
        if (shouldReset)
        {
            cur = 0LL;
            pow = 1LL;

            for (char c : s) cur = (cur * a + (c - 'A')) % p;
            for (int i = 1; i < s.size(); ++i) pow = (pow * a) % p;
            
            shouldReset = false;
        }
        else
        {
            cur = (cur - ((front - 'A') * pow) % p + p) % p;
            cur = (cur * a + (s.back() - 'A')) % p;
        }

        front = s.front();
        return cur;
    }

    static constexpr long long p = 19260817;
    static constexpr long long a = 26;

    mutable bool shouldReset = true;
    mutable long long cur = 0LL;
    mutable long long pow = 1LL;
    mutable char front = '\0';
};

class Solution
{
public:
    std::vector<std::string> findRepeatedDnaSequences(std::string s)
    {
        auto n = static_cast<const int>(s.size());
        
        // No need to reset rolling hash as we only go one pass. 
        // Thus we don't need Hash == const RollingHash &, not passing in hasher. 
        std::unordered_set<std::string_view, RollingHash> hs(40000);
        std::unordered_set<std::string> ans;

        for (int i = 0; i + 9 < n; ++i)
        {
            auto [it, inserted] = hs.insert({s.data() + i, 10});
            if (!inserted) ans.insert(s.substr(i, 10));
        }

        return {ans.cbegin(), ans.cend()};
    }
};