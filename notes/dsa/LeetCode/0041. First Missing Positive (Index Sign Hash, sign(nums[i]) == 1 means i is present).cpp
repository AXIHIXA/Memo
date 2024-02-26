static const int _ = []
{
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);
    std::cout.tie(nullptr);
    std::setvbuf(stdin, nullptr, _IOFBF, 1 << 20);
    std::setvbuf(stdout, nullptr, _IOFBF, 1 << 20);
    return 0;
}();

class Solution 
{
public:
    int firstMissingPositive(std::vector<int> & nums) 
    {
        // Use the input array itself as a hash table. 
        // Except two special cases [0..n - 1] (lacks n) or [1..n] (lacks n + 1), 
        // the answer will lie in [1..n - 1].  
        // First make sure one (smallest positive possible) is present. 
        // 1st pass: Map all non-positives to one. 
        // 2nd pass: For x in nums, negate nums[x] if not yet negated. 
        // 3rd pass: Return 1st positive index. 
        auto n = static_cast<const int>(nums.size());
        if (std::find(nums.cbegin(), nums.cend(), 1) == nums.cend()) return 1;
        for (int & x : nums) if (x <= 0 || n < x) x = 1;
        
        for (int x : nums)
        {
            x = std::abs(x);
            if (x == n) x = 0;
            nums[x] = -std::abs(nums[x]);
        }

        for (int i = 1; i < n; ++i) if (0 < nums[i]) return i;

        return 0 < nums[0] ? n : n + 1;
    }
};