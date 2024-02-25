class Solution
{
public:
    std::string nearestPalindromic(std::string n)
    {
        long long num = std::stoll(n);
        long long a = maxBinSearch(num);
        long long b = minBinSearch(num);
        return std::to_string(std::abs(a - num) <= std::abs(b - num) ? a : b);
    }

    long long maxBinSearch(long long num)
    { 
        long long ans = std::numeric_limits<int>::min();

        for (long long ll = 0LL, rr = num; ll <= rr; )
        {
            long long mi = ll + ((rr - ll) >> 1LL);
            long long palin = overwrite(mi);

            if (palin < num)
            {
                ans = palin; 
                ll = mi + 1;
            }
            else
            {
                rr = mi - 1;
            }
        }

        return ans;
    }

    long long minBinSearch(long long num)
    {
        long long ans = std::numeric_limits<int>::min();

        for (long long ll = num, rr = 1e18; ll <= rr; )
        {
            long long mi = ll + ((rr - ll) >> 1LL);
            long long palin = overwrite(mi);

            if (num < palin)
            {
                ans = palin; 
                rr = mi - 1;
            }
            else
            {
                ll = mi + 1;
            }
        }

        return ans;
    }
    
    // abcde -> abcba, abde -> abba
    long long overwrite(long long num)
    {
        std::string s = std::to_string(num);
        auto n = static_cast<const int>(s.size());
        for (int ll = ((n - 1) >> 1), rr = (n >> 1); 0 <= ll; --ll, ++rr) s[rr] = s[ll];
        return std::stoll(s);
    }
};