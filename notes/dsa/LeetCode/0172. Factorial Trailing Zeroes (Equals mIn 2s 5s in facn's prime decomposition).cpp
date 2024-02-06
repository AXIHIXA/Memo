class Solution
{
public:
    int trailingZeroes(int n)
    {
        // Number of trailing zeros of a Number x
        // equals to the minimum of: 
        //     Number of 2s in its prime decomposition. 
        //     Number of 5s in its prime decomposition. 

        // For n!, 
        // Number of 2s: n/2 + n/4 + n/8 + ... + n/?
        // Number of 5s: n/5 + n/25 + n/125 + ... + n/?
        // The former sum has more items, and each item is also greater, 
        // thus number of 2s in n!'s prime decomposition 
        // must be larger than 5s. 
        
        int ans = 0;
        for (; 5 <= n; n /= 5) ans += n / 5;
        return ans;
    }
};