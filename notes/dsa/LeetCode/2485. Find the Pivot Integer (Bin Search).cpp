class Solution
{
public:
    int pivotInteger(int n)
    {
        auto sum = [](int a, int b) { return (a + b) * (b - a + 1) / 2; };

        int ll = 1;
        int rr = n;

        while (ll <= rr)
        {
            int mi = ll + ((rr - ll) >> 1);
            int s = sum(1, mi);
            int t = sum(mi, n);

            if (s < t) ll = mi + 1;
            else if (t < s) rr = mi - 1;
            else return mi;
        }

        return -1;
    }
};