#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>


class UnionFind
{
public:
    explicit UnionFind(int size) : root(size), rank(size, 0)
    {
        std::iota(root.begin(), root.end(), 0);
    }

    int find(int x)
    {
        if (root[x] == x) return x;
        return root[x] = find(root[x]);
    }

    void unite(int x, int y)
    {
        int rx = find(x);
        int ry = find(y); 
        if (rx == ry) return;

        if (rank[rx] < rank[ry]) root[rx] = ry;
        else if (rank[ry] < rank[rx]) root[ry] = rx;
        else root[ry] = rx, ++rank[rx];
    }

    bool connected(int x, int y)
    {
        return find(x) == find(y);
    }

    void reset(int x)
    {
        // NOT guaranteed safe under all scenarios!
        root[x] = x;
        rank[x] = 0;
    }

    void reset()
    {
        std::iota(root.begin(), root.end(), 0);
        rank.assign(size, 0);
    }

private:
    int size = 0;
    std::vector<int> root;
    std::vector<int> rank;
};


// 快速幂，求 a ** b % p
int pow(int a, int b, int p)
{
    int ans = 1;

    for (; b; b >>= 1)
    {
        if (b & 1) ans = static_cast<long long>(ans) * a % p;
        a = static_cast<long long>(a) * a % p;
    }

    return ans;
}


// 64位整数乘法的 O(log b) 算法
long long mul(long long a, long long b, long long p)
{
    long long ans = 1LL;
    
    for (; b; b >>= 1)
    {
        if (b & 1) ans = (ans + a) % p;
        a = a * 2 % p;
    }

    return ans;
}


// Sieve of Eratosthenes Algorithm. 
std::vector<int> primeDecompose(int x)
{
    std::vector<int> ans;

    for (int factor = 2; factor * factor <= x; )
    {
        if (x % factor == 0)
        {
            x /= factor;
            ans.emplace_back(factor);
        }
        else
        {
            ++factor;
        }
    }

    // Loop ends when x becomes a prime number. 
    // Append this final prime factor. 
    ans.emplace_back(x);

    return ans;
}


// Eucilidean Algorithm. 
// Always replace larger number with remainder large % small
// until remainder == 0, gcd is the remaining non-zero. 
int gcd(int a, int b)
{
    if (a < b) std::swap(a, b);

    while (b)
    {
        int t = b;
        b = a % b;
        a = t;
    }

    return a;
}