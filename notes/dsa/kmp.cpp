#include <iostream>
#include <string>
#include <vector>



std::vector<int> buildProperPrefix(const std::string & s)
{
    // ppx[j]为s[0...j]的真前缀（最长公共前后缀）的长度，包括s[j]
    // ppx[j]由前缀DP求得
    // 求解ppx[j]时，t为ppx[j - 1]
    auto n = static_cast<const int>(s.size());
    std::vector<int> ppx(n, 0);

    for (int t = 0, j = 1; j < n; ++j)
    {
        while (0 < t && s[t] != s[j]) t = ppx[t - 1];
        t = ppx[j] = t + (s[t] == s[j]);
    }

    return ppx;
}


int kmp(const std::string & t, const std::string & p)
{
    auto m = static_cast<const int>(t.size());
    auto n = static_cast<const int>(p.size());

    std::vector<int> ppx = buildProperPrefix(p);

    int i = 0;
    int j = 0;

    while (i < m && j < n)
    {
        if (t[i] == p[j]) ++i, ++j;
        else if (j == 0) ++i;
        else j = ppx[j - 1];
    }

    return j == n ? i - j : -1;
}


// Textbook note hinted by https://dsa.cs.tsinghua.edu.cn/~deng/ds/src_link/
namespace dsacpp
{

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


int kmp(const std::string & target, const std::string & pattern) 
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

}  // namespace dsacpp


int main(int argc, char * argv[])
{
    std::string s = "ABBABAABABAA";
    std::vector<int> ppx = buildProperPrefix(s);
    std::vector<int> next = dsacpp::buildNext(s);

    for (int i = 0; i != s.size(); ++i) 
        std::printf("%4d ", i);
    std::printf("\n");

    for (int i = 0; i != s.size(); ++i) 
        std::printf("%4c ", s[i]);
    std::printf("\n");

    for (int i = 0; i != ppx.size(); ++i) 
        std::printf("%4d ", ppx[i]);
    std::printf("\n");
    
    for (int i = 0; i != next.size(); ++i) 
        std::printf("%4d ", next[i]);
    std::printf("\n");
    
    return EXIT_SUCCESS;
}
