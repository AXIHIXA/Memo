class Solution 
{
public:
    Solution()
    {
        static const int _ = init();
    }

    // Two Pointers. 
    int firstMissingPositive(std::vector<int> & nums) 
    {
        int ll = 0;
        auto rr = static_cast<int>(nums.size());
		
        // [r....]垃圾区
		// 最好的状况下，认为1~r是可以收集全的，每个数字收集1个，不能有垃圾
		// 有垃圾呢？预期就会变差(r--)

		while (ll < rr) 
        {
			if (nums[ll] == ll + 1) 
            {
				ll++;
			} 
            else if (nums[ll] <= ll || rr < nums[ll] || nums[nums[ll] - 1] == nums[ll]) 
            {
				std::swap(nums[ll], nums[--rr]);
			} 
            else 
            {
				std::swap(nums[ll], nums[nums[ll] - 1]);
			}
		}

		return ll + 1;
    }

    // Array as Hash. 
    int firstMissingPositiveArrayAsHash(std::vector<int> & nums) 
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

private:
    static int init()
    {
        std::ios_base::sync_with_stdio(false);
        std::cin.tie(nullptr);
        std::cout.tie(nullptr);
        std::setvbuf(stdin, nullptr, _IOFBF, 1 << 20);
        std::setvbuf(stdout, nullptr, _IOFBF, 1 << 20);
        return 0;
    }
};