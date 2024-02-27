class Solution
{
public:
    // Reverse == Self, might overflow, use long long. 
    bool isPalindrome1(int x)
    {
        if (x < 0) return false;
        if (x < 10) return true;

        // Palindrome: Reverse == Self. 
        // Note: Reverse may overflow so use long long. 
        long long reverse = 0LL;
        int y = x;

        while (y)
        {
            reverse = reverse * 10LL + (y % 10);
            y /= 10;
        }

        return reverse == x;
    }

    // No-overflow version. Special care to 10..00-likes. 
    bool isPalindrome(int x)
    {
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

        int y = 0;

        while (y < x)
        {
            y = y * 10 + (x % 10);
            x /= 10;
        }

        return x == y || x == y / 10;
    }
};

