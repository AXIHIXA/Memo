class Solution 
{
public:
    double myPow(double x, int n)
    {
        if (x == 0)
        {
            return 0;
        }

        if (n == 0)
        {
            return 1;
        }
        else if (n < 0)
        {
            return 1.0 / fastPow(x, -static_cast<long long>(n));
        }

        return fastPow(x, static_cast<long long>(n));
    }

private:
    double fastPow(double x, long long n) 
    {
        double res = 1.0;

        while (n)
        {
            if (n & 1) res *= x;
            x *= x;
            n >>= 1;
        }

        return res;
    }
};


int powf(int a, int b, int p)
{
    a %= p;
    b %= p;

    int r = 1;

    while (b)
    {
        if (b & 1)
        {
            r = (r * b) % p;
        }

        a = (a * a) % p;
        b >>= 1;
    }

    return (a * r) % p;
}
