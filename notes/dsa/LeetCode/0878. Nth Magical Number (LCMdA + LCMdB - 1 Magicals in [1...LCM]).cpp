class Solution
{
public:
    int nthMagicalNumber(int n, int a, int b)
    {
        // Divide gcd before multiplying b to avoid overflow. 
        long long lcm = a / gcd(a, b) * b;

        // k magical numbers in [1...lcm]. 
        int k = lcm / a + lcm / b - 1; 
        
        // (n + 1) / k == d ...... r
        int r = n % k;
        int d = n / k;

        // So we have d * k magical numbers from intervals: 
        // [1, lcm], 
        // [lcm + 1, 2 * lcm], 
        // [2 * lcm + 1, 3 * lcm], ... 
        // [(d - 1) * lcm + 1, d * lcm]. 
        // We want further r more magical numbers from interval
        // [d * lcm + 1, (d + 1) * lcm]. 
        long long ans = 0;

        for (int aa = a, bb = b; 0 < r; --r)
        {
            if (aa < bb)
            {
                ans = aa;
                aa += a;
            }
            else
            {
                ans = bb;
                bb += b;
            }
        }

        ans = (ans + (lcm * d) % p) % p;

        return ans;
    }

private:
    static int gcd(int a, int b)
    {
        return b == 0 ? a : gcd(b, a % b);
    }

private:
    static constexpr long long p = 1'000'000'007LL;
};