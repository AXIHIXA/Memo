// static const int init = []
// {
//     std::ios_base::sync_with_stdio(false);
//     std::cin.tie(nullptr);
//     std::cout.tie(nullptr);
//     std::setvbuf(stdin, nullptr, _IOFBF, 1 << 20);
//     std::setvbuf(stdout, nullptr, _IOFBF, 1 << 20);
//     return 0;
// }();

class Solution 
{
public:
    int findDuplicate(std::vector<int> & nums)
    {
        // return floyds(nums);
        return binarySearch(nums);
    }

private:
    // Treat nums[i] as graph edge (i -> nums[i]), detect cycles. 
    // O(n) time, O(1) space. 
    // No modification on input. 
    static int floyds(const std::vector<int> & nums)
    {
        int slow = nums[0];
        int fast = nums[0];

        do 
        {
            slow = nums[slow];
            fast = nums[nums[fast]];
        }
        while (slow != fast);

        slow = nums[0];

        while (slow != fast)
        {
            slow = nums[slow];
            fast = nums[fast];
        }

        return slow;
    }

    // Use sign bit of nums[x] as occurance of x. 
    // O(n) time, O(1) space. 
    // Modifies input but could restore. 
    static int negativeMarking(std::vector<int> & nums) 
    {
        int ans = 0;

        for (int x : nums) 
        {
            int xp = std::abs(x);
            if (nums[xp] < 0) { ans = xp; break; }
            nums[xp] *= -1;
        }

        for (int & x : nums) x = std::abs(x);

        return ans;
    }

    // Array as hashmap, let nums[i] store i. 
    // O(n) time, constant space. 
    // Modifies input!
    static int cyclicSort(std::vector<int> & nums)
    {
        while (nums[0] != nums[nums[0]])
        {
            std::swap(nums[0], nums[nums[0]]);
        }

        return nums[0];
    }

    // Count total number of 1-bits for range [1, n], compare with that for nums, 
    // the duplicate is bits where bc1n < bcNums. 
    // O(nlogn) time, O(1) space. 
    // No modification on input. 
    static int bitCount(const std::vector<int> & nums)
    {
        auto n = static_cast<int>(nums.size()) - 1;
        if (n == 1) return nums[0];
        
        int maxNum = *std::max_element(nums.cbegin(), nums.cend());
        int maxBit = 0;
        while (maxNum) ++maxBit, (maxNum >>= 1);

        int ans = 0;

        for (int i = 0, mask = 1; i < maxBit; ++i, mask <<= 1)
        {
            int bc1n = 0;
            int bcNums = 0;
            for (int i = 1; i <= n; ++i) bc1n += ((i & mask) != 0);
            for (int x : nums) bcNums += ((x & mask) != 0);
            if (bc1n < bcNums) ans |= mask;
        }

        return ans;
    }

    // Binary search, split w.r.t. mi v.s. sum([t <= nums[mi] for t in nums]). 
    // O(nlogn) time, O(1) space. 
    // No modification on input. 
    static int binarySearch(const std::vector<int> & nums)
    {
        int ans = 0;

        int lo = 1, hi = nums.size();

        while (lo <= hi)
        {
            int mi = lo + ((hi - lo) >> 1);
            int le = std::count_if(
                    nums.cbegin(), 
                    nums.cend(), 
                    [t = nums[mi]](int x) { return x <= t; }
            );

            if (mi < le)
            {
                ans = mi;
                hi = mi - 1;
            }
            else
            {
                lo = mi + 1;
            }
        }

        return ans;
    }
};
