class Solution
{
public:
    Solution() 
    {
        static int init = collect();
    }    

    int superpalindromesInRange(std::string left, std::string right)
    {
        // std::numeric_limits<long long>::max() == 9223372036854775807, 20 digits. 
        // 1 <= left.size(), right.size() <= 18. 

        return std::upper_bound(table.cbegin(), table.cend(), std::stoll(right)) - 
               std::lower_bound(table.cbegin(), table.cend(), std::stoll(left));
    }

private:
    static int collect()
    {
        const long long maxi = std::sqrt(std::numeric_limits<long long>::max()) + 1LL;
        
        for (long long x = 1; ; ++x)
        {
            long long p = generate(x, false);
            if (maxi < p) break;
            long long p2 = p * p;
            if (isPalindrome(p2)) table.emplace_back(p2);
        }

        for (long long x = 1; ; ++x)
        {
            long long p = generate(x, true);
            if (maxi < p) break;
            long long p2 = p * p;
            if (isPalindrome(p2)) table.emplace_back(p2);
        }

        std::sort(table.begin(), table.end());

        return 0;
    }

    static bool isPalindrome(long long x)
    {
        if (x < 0LL || (x % 10LL == 0LL && x != 0LL)) return false;
        if (x < 10LL) return true;

        long long reverse = 0LL;

        while (reverse < x)
        {
            reverse = reverse * 10LL + x % 10LL;
            x /= 10LL;
        }

        return reverse == x || reverse / 10LL == x;
    }

    static long long generate(long long x, bool duplicateBack)
    {
        long long ans = x;
        if (!duplicateBack) ans /= 10LL;

        while (x)
        {
            ans = ans * 10LL + x % 10LL;
            x /= 10LL;
        }

        return ans;
    }

private:
    static std::vector<long long> table;
};

std::vector<long long> Solution::table;
