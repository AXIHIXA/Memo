class Solution
{
public:
    bool isHappy(int n)
    {
        if (n == 1) return true;
        if (n == 2) return false;
        
        int slow = n;
        int fast = next(next(slow));

        while (fast != 1 && fast != slow)
        {
            slow = next(slow);
            fast = next(next(fast));
        }

        return fast == 1;
    }

private:
    static int next(int n)
    {
        int ans = 0;

        while (n)
        {
            int d = n % 10;
            ans += d * d;
            n /= 10;
        }

        return ans;
    }
};