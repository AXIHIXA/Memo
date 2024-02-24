class Solution
{
public:
    int twoEggDrop(int n)
    {
        // dp[i][j]: Max floors solvable with i moves and j eggs. 
        std::vector dp(n + 1, std::vector<int>(3, 0));
        
        for (int i = 1; i <= n; ++i)
        {
            for (int j = 1; j <= 2; ++j)
            {
                dp[i][j] = 1 + dp[i - 1][j] + dp[i - 1][j - 1];
                if (n <= dp[i][j]) return i;
            }
        }

        return -1;
    }

    int twoEggDropO1(int n)
    {
        // https://leetcode.com/problems/egg-drop-with-2-eggs-and-n-floors/solutions/1246621/java-o-1-intuition-and-detailed-reasoning-100-time-100-space/
        
        // Suppose we partition floors into equal-sized chunks (each of size c + 1), 
        // and try at the highest floor of each partition. 
        // If egg breaks at p-th partition, 
        // we need to try c floors, for p + c tries in total. 
        // If breaks at (p + 1)-th partition, 
        // we need p + 1 + c tries in total. 

        // If we make (p + 1)-th partition 1-floor smaller than p-th partition, 
        // then we could still have p + c tries on a (p + 1)-th-partition break. 
        // I.e., we partition into chunks of size x, x - 1, x - 2, ... 1, 
        // and the answer will be exactly x. 

        // Thus we need to know a smallest number x s.t. 
        // x + (x - 1) + (x - 2) + ... + 1 >= n, 
        // which is ceil of the positive solution for x**2 + x - 2n = 0. 
        return std::ceil((-1.0 + std::sqrt(1 + 8 * n)) / 2.0); 
    }

    int twoEggDropOn(int n)
    {
        // O(n) version to calculate x in the O(1) solution above.  
        int ans = 0;
        for (int i = 1; 0 < n; ++i, ++ans) n -= i;
        return ans;
    }
};