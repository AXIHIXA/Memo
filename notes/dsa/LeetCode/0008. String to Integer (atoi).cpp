class Solution
{
public:
    int myAtoi(std::string s)
    {
        auto n = static_cast<const int>(s.size());

        int sign = 0;
        bool couldHaveSign = true;
        long long ans = 0LL;

        int i = 0;
        while (i < n && s[i] == ' ') ++i;

        for ( ; i < n; ++i)
        {
            if (couldHaveSign)
            {
                if (s[i] == '-')
                {
                    sign = -1;
                    couldHaveSign = false;
                    continue;
                }

                if (s[i] == '+')
                {
                    sign = 1;
                    couldHaveSign = false;
                    continue;
                }
            }

            if (s[i] != ' ' && s[i] != 0) couldHaveSign = false;
            
            if (std::isdigit(s[i]))
            {
                ans = ans * 10LL;

                if (kLimit <= ans)
                {
                    ans = kLimit;
                    break;
                }

                ans = ans + (s[i] - '0');

                if (kLimit <= ans)
                {
                    ans = kLimit;
                    break;
                }

                continue;
            }
            
            break;
        }
        
        if (sign == 0) sign = 1;

        return sign == 1 ? std::min(ans, kIntMax) : -ans;
    }

private:
    static constexpr long long kIntMax = std::numeric_limits<int>::max();
    static constexpr long long kIntMin = std::numeric_limits<int>::min();
    static constexpr long long kLimit = 2147483648LL;
};