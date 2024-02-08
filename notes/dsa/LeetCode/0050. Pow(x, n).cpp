class Solution
{
public:
    double myPow(double x, int n)
    {
        if (x == 0) return 0;
        if (n < 0) return 1.0 / powf(x, -static_cast<long long>(n));
        return powf(x, n);
    }

private:
    double powf(double x, long long n)
    {
        double ans = 1.0;

        for (; n; n >>= 1)
        {
            if (n & 1) ans *= x;
            x *= x;
        }

        return ans;
    }
};