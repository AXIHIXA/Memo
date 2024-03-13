class Solution
{
public:
    bool isPalindrome(int x)
    {
        // Special care to 10..00-likes. 
        if (x < 0 || (x % 10 == 0 && x != 0)) return false;
        if (x < 10) return true;

        // Odd-digit palindrome: 
        // x       y
        // 23432   0
        // 2343    2
        // 234     23
        // 23      234

        // Even-digit palindrome:
        // x       y
        // 234432  0
        // 23443   2
        // 2344    23
        // 234     234

        // Edge case: x % 10 == 0
        // x       y
        // 10      0
        // 1       0
        // 0       1

        int reverse = 0;

        while (reverse < x)
        {
            reverse = reverse * 10 + x % 10;
            x /= 10;
        }

        return reverse == x || reverse / 10 == x;
    }
};

