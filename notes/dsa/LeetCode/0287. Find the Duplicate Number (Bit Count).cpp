class Solution 
{
public:
    int findDuplicate(std::vector<int> & nums) 
    {
        std::ios_base::sync_with_stdio(false);
        std::cin.tie(NULL);
        
        int n = nums.size() - 1;
        int msb = mostSignificantBit(*std::max_element(nums.cbegin(), nums.cend()));
        
        int ans = 0;

        for (int bit = 0; bit != msb; ++bit)
        {
            int baseCount = 0;
            int numsCount = 0;
            int mask = (1 << bit);

            for (int i = 0; i <= n; ++i)
            {
                baseCount += ((i & mask) != 0);
                numsCount += ((nums[i] & mask) != 0);
            }

            if (baseCount < numsCount)
            {
                ans |= mask;
            }
        }

        return ans;
    }

private:
    static int mostSignificantBit(int num)
    {
        int i = 0;

        while (num)
        {
            num >>= 1;
            ++i;
        }

        return i;
    }
};
