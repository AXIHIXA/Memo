class Solution
{
public:
    bool isPalindrome(int x)
    {
        if (x < 0 || (x % 10 == 0 && x != 0)) return false;
        if (x < 10) return true;
        
        int inverse = 0;

        while (inverse < x)
        {
            inverse *= 10;
            inverse += x % 10;
            x /= 10;
        }

        return x == inverse || x == inverse / 10;
    }
};