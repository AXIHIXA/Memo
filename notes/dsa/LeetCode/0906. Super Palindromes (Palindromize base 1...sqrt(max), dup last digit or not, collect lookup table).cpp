class Solution
{
public:
    Solution()
    {
        static int _ = collect();
    }

    int superpalindromesInRange(std::string left, std::string right)
    {
        // 1e18 - 1 < std::numeric_limits<long long>::max();
        int ans = 0;
        auto b = std::upper_bound(table.cbegin(), table.cend(), std::stoll(left), std::less_equal<long long>());
        auto e = std::upper_bound(table.cbegin(), table.cend(), std::stoll(right));
        return e - b;
    }

private:
    static int collect()
    {
        static auto rr = static_cast<long long>(std::sqrt(std::numeric_limits<long long>::max()));

        for (int base = 1; ; ++base)
        {
            long long p1 = palindromize(base, false);
            if (rr < p1) break;
            long long sp1 = p1 * p1;
            if (isPalindrome(sp1)) table.emplace_back(sp1);
        }

        for (int base = 1; ; ++base)
        {
            long long p2 = palindromize(base, true);
            if (rr < p2) break;
            long long sp2 = p2 * p2;
            if (isPalindrome(sp2)) table.emplace_back(sp2);
        }

        std::sort(table.begin(), table.end());

        return static_cast<int>(table.size());
    }

    static long long palindromize(int base, bool duplicateLastDigit)
    {
        long long prefix = base;
        if (!duplicateLastDigit) prefix /= 10LL;

        long long reverse = 0LL;

        for (int x = base; 0 < x; x /= 10)
        {
            prefix *= 10LL;
            reverse = reverse * 10LL + (x % 10);
        }

        return prefix + reverse;
    }

    static long long isPalindrome(long long x)
    {
        if (x < 0LL || x % 10LL == 0LL && x != 0LL) return false;
        if (x < 10LL) return true;

        long long y = 0;
        
        while (y < x)
        {
            y = y * 10LL + (x % 10LL);
            x /= 10LL;
        }

        return x == y || x == y / 10LL;
    }

    static std::vector<long long> table;
};

std::vector<long long> Solution::table;
