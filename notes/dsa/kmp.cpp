#include <iostream>
#include <string>
#include <vector>

std::vector<int> buildNext(const std::string & p)
{
    auto m = static_cast<const int>(p.size());
    std::vector<int> next(m, -1);

    // next[j]为p[:j]的最长公共前后缀 (Proper Prefix) 的长度
    // next[j]由前缀DP求得
    // 求解next[j]时，t为next[j - 1]
    for (int t = -1, j = 0; j < m - 1; )
    {
        if (t < 0 || p[t] == p[j])
        {
            ++t, ++j;
            next[j] = p[t] == p[j] ? next[t] : t;
        }
        else
        {
            t = next[t];  
        }
    }
    
    return next;
}

std::vector<int> solveProperPrefix(const std::string & s)
{
    // dp[j]为s[:j+1]的最长公共前后缀的长度 (with OPTIMIZATIONS!)
    // dp[j]由前缀DP求得
    // 求解dp[j]时，t为dp[j - 1]
    auto m = static_cast<const int>(s.size());
    std::vector<int> dp(m, 0);

    for (int t = 0, j = 1; j < m; ++j)
    {
        while (0 < t && s[t] != s[j]) t = dp[t - 1];
        t = dp[j] = t + (s[t] == s[j]);
    }

    return dp;
}

int kmp(const std::string & pattern, const std::string & target) 
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

    return i - j;
}

int main()
{
    std::string s = "ABBABAABABAA";
    std::vector<int> next = buildNext(s);
    std::vector<int> ppl = solveProperPrefix(s);

    for (int i = 0; i != s.size(); ++i) 
        std::printf("%4d ", i);
    std::printf("\n");

    for (int i = 0; i != s.size(); ++i) 
        std::printf("%4c ", s[i]);
    std::printf("\n");
    
    for (int i = 0; i != next.size(); ++i) 
        std::printf("%4d ", next[i]);
    std::printf("\n");

    for (int i = 0; i != ppl.size(); ++i) 
        std::printf("%4d ", ppl[i]);
    std::printf("\n");
    
    return EXIT_SUCCESS;
}
