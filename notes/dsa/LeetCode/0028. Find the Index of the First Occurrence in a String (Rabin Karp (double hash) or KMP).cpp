class Solution
{
public:
    int strStr(string haystack, string needle)
    {
        return rabinKarp(needle, haystack);
    }

private:
    static constexpr long long kRadix1 = 26LL;
    static constexpr long long kPrime1 = 1000000033LL;
    static constexpr long long kRadix2 = 27LL;
    static constexpr long long kPrime2 = 2147483647LL;

    static int rabinKarp(const std::string & pattern, const std::string & target)
    {
        int m = target.size();
        int n = pattern.size();
        if (m < n) return -1;

        // Use double hash to reduce hash conflicts. 
        long long maxWeight1 = 1LL;
        long long maxWeight2 = 1LL;

        for (int i = 0; i != n; ++i)
        {
            maxWeight1 = (maxWeight1 * kRadix1) % kPrime1;
            maxWeight2 = (maxWeight2 * kRadix2) % kPrime2;
        }

        auto [hp1, hp2] = hash(pattern.c_str(), n);
        auto [ht1, ht2] = hash(target.c_str(), n);

        for (int i = 0; i <= m - n; ++i)
        {
            if (0 < i)
            {
                ht1 = (
                    (ht1 * kRadix1) % kPrime1 - 
                    (static_cast<long long>(target[i - 1] - 'a') * maxWeight1) % kPrime1 + 
                    static_cast<long long>(target[i + n - 1] - 'a') +
                    kPrime1
                ) % kPrime1;

                ht2 = (
                    (ht2 * kRadix2) % kPrime2 -
                    (static_cast<long long>(target[i - 1] - 'a') * maxWeight2) % kPrime2 + 
                    static_cast<long long>(target[i + n - 1] - 'a') + 
                    kPrime2
                ) % kPrime2;
            }

            // If the hash matches, return immediately. 
            // Probability of hash conflicts tends to zero. 
            if (hp1 == ht1 && hp2 == ht2)
            {
                return i;
            }
        }

        return -1;
    }

    static std::pair<long long, long long> hash(const char * s, int hi)
    {
        long long h1 = 0LL;
        long long h2 = 0LL;
        long long f1 = 1LL;
        long long f2 = 1LL;

        for (int i = hi - 1; 0 <= i; --i)
        {
            h1 = (h1 + (static_cast<long long>(s[i] - 'a') * f1) % kPrime1) % kPrime1;
            f1 = (f1 * kRadix1) % kPrime1;
            h2 = (h2 + (static_cast<long long>(s[i] - 'a') * f2) % kPrime2) % kPrime2;
            f2 = (f2 * kRadix2) % kPrime2;
        }

        return {h1, h2};
    }

    static int kmp(const std::string & pattern, const std::string & target) 
    {
        int m = target.size(), i = 0;
        int n = pattern.size(), j = 0;
        if (m < n) return -1;

        std::vector<int> next = buildNext(pattern);

        while (j < n && i < m)
        {
            if (j < 0 || target[i] == pattern[j]) ++i, ++j;
            else                                  j = next[j];
        }

        return j == n ? i - j : -1;
    }

    static std::vector<int> buildNext(const std::string & pattern)
    {
        int m = pattern.size();
        std::vector<int> next(m, -1);

        // next[j]为pattern[:j]的最长公共前后缀的长度
        // next[j]由前缀DP求得
        // 求解next[j]时，t为next[j - 1]
        for (int t = -1, j = 0; j < m - 1; )
        {
            if (t < 0 || pattern[t] == pattern[j])
            {
                ++t, ++j;
                next[j] = pattern[t] == pattern[j] ? next[t] : t;
            }
            else
            {
                t = next[t];  
            }
        }
        
        return next;
    }
};
