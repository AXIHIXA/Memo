class Solution
{
public:
    int reverse(int x)
    {
        int ans = 0;
        
        for ( ; x != 0; x /= 10)
        {
            if (ans < kIntMin / 10 || kIntMax / 10 < ans) return 0;
            ans = ans * 10 + (x % 10);
        }

        return ans;
    }

private:
    static constexpr int kIntMax = std::numeric_limits<int>::max();
    static constexpr int kIntMin = std::numeric_limits<int>::min();
};