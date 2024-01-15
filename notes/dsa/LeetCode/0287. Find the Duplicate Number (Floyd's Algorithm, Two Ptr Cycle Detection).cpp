class Solution 
{
public:
    int findDuplicate(vector<int> & nums) 
    {
        std::ios_base::sync_with_stdio(false);
        std::cin.tie(NULL);
        
        // Floyd's algorithm (Two pointers). 
        int slow = nums[nums[0]];
        int fast = nums[nums[nums[0]]];

        // Suppose the length to loop entrance is F, 
        // and the length of the loop is C. 
        // When fast meets slow at location a in loop, 
        // dist(fast) == F + nC + a == 2 * dist(slow)
        // dist(slow) == F + a
        // Thus a == nC - F. 
        while (slow != fast)
        {
            slow = nums[slow];
            fast = nums[nums[fast]];
        }

        // Restart slow and march both pointers
        // at pace of slow, they will meet
        // in F steps, i.e., at the loop entrance. 
        slow = nums[0];

        while (slow != fast)
        {
            slow = nums[slow];
            fast = nums[fast];
        }

        return fast;
    }
};