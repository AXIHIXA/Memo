class Solution 
{
public:
    int fib(int n) 
    {
        if (n == 0)
        {
            return 0;
        }
        else if (n == 1)
        {
            return 1;
        }

        n -= 1;
        
        int a = 0;
        int b = 1;

        while (n--)
        {
            b = a + b;
            a = b - a;
        }   

        return b;
    }
};
