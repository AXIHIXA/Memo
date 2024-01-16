class Solution 
{
public:
    void rotate(vector<int> & nums, int k) 
    {
        std::ios_base::sync_with_stdio(false);
        std::cin.tie(nullptr);
        std::cout.tie(nullptr);
        
        k %= nums.size();
        if (k == 0) return;

        // Option 1: Cyclic Replacement.
        for (int i = 0, count = 0, n = nums.size(); count < n; ++i)
        {
            int cur = i;
            int prev = nums[i];

            do
            {
                int next = (cur + k) % n;
                int tmp = nums[next];
                nums[next] = prev;
                prev = tmp;
                cur = next;
                ++count;
            }
            while (cur != i);
        }

        // Option 2: By reverse. 
        // std::reverse(nums.begin(), nums.end() - k);
        // std::reverse(nums.end() - k, nums.end());
        // std::reverse(nums.begin(), nums.end());
    }
};