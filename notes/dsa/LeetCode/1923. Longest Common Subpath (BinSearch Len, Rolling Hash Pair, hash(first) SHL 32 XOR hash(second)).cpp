class RollingHash
{
public:
    std::size_t operator()(const std::vector<int> & vec, int b, int l) const
    {
        if (resetFlag)
        {
            resetFlag = false;

            bp1 = 1LL;
            bp2 = 1LL;

            for (int i = 1; i < l; ++i)
            {
                bp1 = (bp1 * b1) % p1;
                bp2 = (bp2 * b2) % p2;
            }

            c1 = 0LL;
            c2 = 0LL;

            for (int i = 0; i < l; ++i)
            {
                c1 = ((c1 * b1) % p1 + vec.at(b + i)) % p1;
                c2 = ((c2 * b2) % p2 + vec.at(b + i)) % p2;
            }
        }
        else
        {
            c1 = (c1 - (front * bp1) % p1 + p1) % p1;
            c1 = ((c1 * b1) % p1 + vec.at(b + l - 1)) % p1;
            c2 = (c2 - (front * bp2) % p2 + p2) % p2;
            c2 = ((c2 * b2) % p2 + vec.at(b + l - 1)) % p2;
        }

        // std::printf("Rolling Hash for [ ");
        // for (int i = b; i < b + l; ++i) std::printf("%d ", vec.at(i));
        // std::printf("] is (%lld, %lld) -> (%lu, %lu) -> %lu\n", c1, c2, llh(c1), llh(c2), (llh(c1) << 32) ^ llh(c2));

        front = vec.at(b);

        return (llh(c1) << 32) ^ llh(c2);
    }

    void reset()
    {
        resetFlag = true;
    }

private:
    static constexpr long long p1 = 1'000'000'007LL;
    static constexpr long long p2 = 1'000'000'009LL;

    static constexpr long long b1 = 100'007LL;
    static constexpr long long b2 = 100'009LL;

    static std::hash<long long> llh;

    mutable long long c1 = 0LL;
    mutable long long c2 = 0LL;

    mutable long long bp1 = 1LL;
    mutable long long bp2 = 1LL;

    mutable int front = 0;

    mutable long long resetFlag = true;
};

std::hash<long long> RollingHash::llh;

class Solution
{
public:
    int longestCommonSubpath(int n, std::vector<std::vector<int>> & paths)
    {
        auto m = static_cast<const int>(paths.size());

        int lo = 1;
        int hi = std::min_element(
                paths.cbegin(), 
                paths.cend(), 
                [](const auto & a, const auto & b) 
                { 
                    return a.size() < b.size(); 
                }
        )->size();
        int ans = 0;
        RollingHash rh;

        while (lo <= hi)
        {
            int len = lo + ((hi - lo) >> 1);

            std::unordered_set<std::size_t> s;
            bool check = true;

            for (int i = 0; i < m; ++i)
            {
                std::unordered_set<std::size_t> t;
                rh.reset();

                for (int j = 0; j + len - 1 < paths[i].size(); ++j)
                {
                    std::size_t hash = rh(paths[i], j, len);

                    if (i == 0 || s.contains(hash))
                    {
                        t.emplace(hash);
                    }
                }

                if (t.empty())
                {
                    check = false;
                    break;
                }

                s = std::move(t);
            }

            if (check)
            {
                ans = len;
                lo = len + 1;
            }
            else
            {
                hi = len - 1;
            }
        }

        return ans;
    }
};