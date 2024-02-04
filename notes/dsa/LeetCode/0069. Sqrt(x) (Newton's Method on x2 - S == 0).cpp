class Solution 
{
public:
    int mySqrt(int x)
    {
        if (x < 2) return x;

        // Newton's method: x1 = x0 - (f(x0) / f'(x0)). 
        // f(x) = x**2 - S, x1 = x0 + 0.5 (x0 + S / x0). 
        double x0 = x;
        double x1 = 0.5 * (x0 + x / x0);

        while (1 <= std::abs(x0 - x1))
        {
            x0 = x1;
            x1 = 0.5 * (x0 + x / x0);
        }

        return static_cast<int>(x1);
    }
};