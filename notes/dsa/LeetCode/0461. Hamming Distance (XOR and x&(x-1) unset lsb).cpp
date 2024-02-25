class Solution
{
public:
    int hammingDistance(int x, int y)
    {
        int z = x ^ y;
        int ans = 0;
        while (0 < z) z &= (z - 1), ++ans;
        return ans;
    }
};