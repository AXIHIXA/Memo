#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>


class UnionFind
{
public:
    explicit UnionFind(int sz) : root(sz), rank(sz, 1)
    {
        std::iota(root.begin(), root.end(), 0);
    }

    int find(int x)
    {
        if (x == root[x]) return x;
        return root[x] = find(root[x]);
    }

    void merge(int x, int y)
    {
        if (int rx = find(x), ry = find(y); rx != ry)
        {
            if (rank[rx] < rank[ry]) root[rx] = ry;
            else if (rank[ry] < rank[rx]) root[ry] = rx;
            else { root[ry] = rx; ++rank[rx]; }
        }
    }

    bool connected(int x, int y)
    {
        return find(x) == find(y);
    }

private:
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
