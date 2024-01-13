class Solution 
{
public:
    int minOperations(int n) 
    {
        int ans = 0;
        
        while (n)
        {
            switch (n & 3)
            {
                case 0:
                {
                    n >>= 2;
                    break;
                }
                case 1:
                {
                    n >>= 1;
                    ++ans;
                    break;
                }
                case 2:
                {
                    n >>= 1;
                    break;
                }
                case 3: 
                {
                    ++n, ++ans;
                    n >>= 2;
                    break;
                }
                default:
                {
                    throw runtime_error("");
                }
            }
        }

        return ans;
    }
};