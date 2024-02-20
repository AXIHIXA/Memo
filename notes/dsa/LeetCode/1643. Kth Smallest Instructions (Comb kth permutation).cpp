class Solution
{
public:
    std::string kthSmallestPath(std::vector<int> & destination, int k)
    {
        int h = destination[1];
        int v = destination[0];
        const int n = h + v;

        std::string ans;
        ans.reserve(n);

        for (int i = 0; i < n; ++i)
        {
            if (h)
            {
                int c = comb(h - 1 + v, v);

                if (k <= c)
                {
                    ans += 'H';
                    --h;
                }
                else
                {
                    k -= c;
                    ans += 'V';
                    --v;
                }
            }
            else
            {
                ans += 'V';
                --v;
            }
        }

        return ans;
    }

    int comb(int n, int r)
    {
        int ans = 1;

        for (int i = 1, j = n - r + 1; i <= r; ++i, ++j)
        {
            ans = ans * j / i;
        }

        return ans;
    }
};