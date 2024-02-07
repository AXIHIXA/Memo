class Solution
{
public:
    int singleNumber(std::vector<int> & nums)
    {
        std::array<int, 32> mask {0};

        for (int x : nums) 
        {
            for (int i = 0; i < 32; ++i)
            {
                mask[i] += ((x >> i) & 1);
            }
        }

        int ans = 0;
        
        for (int i = 0; i < 32; ++i)
        {
            if (mask[i] % 3)
            {
                ans |= (1 << i);
            }
        }

        return ans;
    }
};