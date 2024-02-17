static inline bool check(int x, int target)
{
    if (x == target) return true;
    
    for (int divisor = 10; divisor <= x && x % divisor <= target; divisor *= 10)
    {
        if (check(x / divisor, target - (x % divisor))) return true;
    }

    return false;
}

static std::array<int, 1001> f {0, 1};

static const int generateLookupTable = []
{
    for (int i = 2; i <= 1000; ++i)
    {
        f[i] = f[i - 1];
        if (check(i * i, i)) f[i] += i * i;
    }
    
    return 0;
}();

class Solution
{
public:
    int punishmentNumber(int n)
    {
        return f[n];
    }
};