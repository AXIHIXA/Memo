class HashMap
{
public:
    using PrefixSum = int;
    using LeftmostIndex = int;

public:
    int findTheLongestSubstring(std::string s)
    {
        auto n = static_cast<const int>(s.size());

        int ans = 0;
        std::unordered_map<PrefixSum, LeftmostIndex> hashMap;
        hashMap.emplace(0, -1);

        for (int i = 0, ps = 0; i < n; ++i)
        {
            ps ^= weight(s[i]);
            auto it = hashMap.find(ps);

            if (it != hashMap.end())
            {
                ans = std::max(ans, i - it->second);
            }

            if (hashMap.count(ps) == 0)
            {
                hashMap.emplace(ps, i);
            }
        }

        return ans;
    }

private:
    inline int weight(char c)
    {
        switch (c)
        {
            case 'a':
                return 1;
            case 'e':
                return 2;
            case 'i':
                return 4;
            case 'o':
                return 8;
            case 'u':
                return 16;
            default:
                return 0;
        }
    }
};

class Bucket
{
public:
    int findTheLongestSubstring(std::string s)
    {
        auto n = static_cast<const int>(s.size());

        int ans = 0;
        std::array<int, 32> hashMap;
        hashMap[0] = -1;  // prefix sum mask, leftmost index. 
        std::fill_n(hashMap.begin() + 1, 31, -2);

        for (int i = 0, ps = 0; i < n; ++i)
        {
            ps ^= weight[s[i] - 'a'];

            if (hashMap[ps] != -2)
            {
                ans = std::max(ans, i - hashMap[ps]);
            }

            if (hashMap[ps] == -2)
            {
                hashMap[ps] = i;
            }
        }

        return ans;
    }

private:
    static constexpr std::array<int, 26> weight = 
    {1,0,0,0,2,0,0,0,4,0,0,0,0,0,8,0,0,0,0,0,16,0,0,0,0,0};
};

// using Solution = HashMap;
using Solution = Bucket;
