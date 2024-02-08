class Solution
{
public:
    int rangeBitwiseAnd(int left, int right)
    {
        // // Brian Kernighan's algorithm. 
        // while (left < right) right &= (right - 1);
        // return left & right;

        // Common binary prefix of two numbers. 
        int shift = 0;
        while (left < right) left >>= 1, right >>= 1, ++shift;
        return left << shift;
    }
};