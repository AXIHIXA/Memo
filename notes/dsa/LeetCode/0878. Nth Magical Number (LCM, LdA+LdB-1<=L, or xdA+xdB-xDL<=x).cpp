class Solution
{
public:
    int nthMagicalNumber(int n, int a, int b)
    {
        long long L = a / gcd(a, b) * b;

        // f(x): Number of magical numbers <= x. 
        auto f = [a, b, L](long long x)
        { 
            return x / a + x / b - x / L;
        };

        long long M = f(L);

        long long lo = 0LL;
        long long hi = L * ((n + M - 1) / M);

        while (lo < hi)
        {
            long long mi = lo + ((hi - lo) >> 1);

            if (f(mi) < n) lo = mi + 1;
            else           hi = mi;
        }

        return lo % p;
    }
    
    int nthMagicalNumberMath(int n, int a, int b)
    {
        // [a, b] == a * b / (a, b). 
        // 1. There are M = L / a + L / b - 1 magical numbers <= L. 
        // 2. If X <= L is magical, then X + L is also magical. 
        // 3. n = M * q + r (r < M), the first L * q numbers contain M * q magical numbers. 
        // 4. Within [M * q + 1, M * q + q] we want to find r - 1 remaining magical numbers. 
        int L = a / gcd(a, b) * b;
        int M = L / a + L / b - 1;
        int q = n / M;
        int r = n % M;

        long long ans = (static_cast<long long>(q) * L) % p;
        if (!r) return ans;
        
        // Find the trailing r - 1 magical numbers. 
        // Offset [M * q + 1, M * q + q] into [1, q], then 
        // the first number x is either a or b, 
        // and the second number is x + a or x + b, etc. 
        int ll = a;
        int rr = b;

        for (int i = 0; i < r - 1; ++i)
        {
            if (ll <= rr) ll += a;
            else          rr += b;
        }

        ans = (ans + std::min(ll, rr)) % p;

        return ans;
    }

private:
    static constexpr int p = 1'000'000'007;

    static int gcd(int a, int b)
    {
        if (a < b) std::swap(a, b);

        for (int t; b; )
        {
            t = a % b;
            a = b;
            b = t;
        }

        return a;
    }
};