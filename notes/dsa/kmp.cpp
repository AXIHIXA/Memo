#include <iostream>
#include <string>
#include <vector>

std::vector<int> buildNext(const std::string & pattern)
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
    std::string pattern = "ABBABAABABAA";
    std::vector<int> next = buildNext(pattern);

    for (int i = 0; i != pattern.size(); ++i) 
        std::printf("%4d ", i);
    std::printf("\n");

    for (int i = 0; i != pattern.size(); ++i) 
        std::printf("%4c ", pattern[i]);
    std::printf("\n");
    
    for (int i = 0; i != next.size(); ++i) 
        std::printf("%4d ", next[i]);
    std::printf("\n");
    
    return EXIT_SUCCESS;
}
