#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>


int partition(int * a, int lo, int hi)
{
    // a[lo, hi]
    std::swap(a[lo], a[lo + std::rand() % (hi - lo + 1)]);
    int pivot = a[lo];

    // if (lo < hi) enforced in all modfications of lo and hi, s.t. lo == hi after this while
    while (lo < hi)
    {
        // Version 2: Swap-first, good when lots of dup elements
        while (lo < hi && pivot < a[hi]) --hi;
        if (lo < hi) a[lo++] = a[hi];
        while (lo < hi && a[lo] < pivot) ++lo;
        if (lo < hi) a[hi--] = a[lo];
    }

    // assert(lo == hi);
    a[lo] = pivot;
    return lo;
}


void sort(int * a, int lo, int hi)
{
    // a[lo, hi)
    if (hi - lo < 2) return;

    int mi = partition(a, lo, hi - 1);
    sort(a, lo, mi);
    sort(a, mi + 1, hi);
}


int quickSelect(std::vector<int> & a, int k)
{
    for (int lo = 0, hi = static_cast<int>(a.size()) - 1; lo < hi; )
    {
        int i = lo;
        int j = hi;
        int pivot = a[lo];

        while (i < j)
        {
            while (i < j and pivot < a[j]) --j;
            if (i < j) a[i++] = a[j];
            while (i < j and a[i] < pivot) ++i;
            if (i < j) a[j--] = a[i];
        }

        a[i] = pivot;

        if (k <= i) hi = i - 1;
        if (i <= k) lo = i + 1;
    }

    return a[k];
}


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


/// (a * b) % p == ((a % p) * (b % p)) % p
long long fpow(long long a, long long b, long long p)
{
    long long r = 1;
    a %= p;

    while (b)
    {
        if (b & 1) r = (r * a) % p;
        a = (a * a) % p;
        b >>= 1;
    }

    return r;
}

