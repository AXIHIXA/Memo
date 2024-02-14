class Solution
{
public:
    int minimizeXor(int num1, int num2)
    {
        int bc = binCount(num2);
        int ans = 0;

        // Turn off as many bits as possible from left to right. 
        for (int i = 31; 0 <= i && 0 < bc; --i)
        {
            if ((num1 >> i) & 1)
            {
                ans |= (1 << i);
                --bc;
            }
        }

        // Remaining non-zero bits are appended to right to minimize the xor result. 
        for (int i = 0; i < 32 && 0 < bc; ++i)
        {
            if (!((ans >> i) & 1))
            {
                ans |= (1 << i);
                --bc;
            }
        }

        return ans;
    }

private:
    static int binCount(int x)
    {
        int ans = 0;
        while (x) ++ans, x &= (x - 1);
        return ans;
    }
};