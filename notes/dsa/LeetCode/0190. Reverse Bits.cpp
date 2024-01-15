class Solution 
{
public:
    uint32_t reverseBits(uint32_t n) 
    {
        uint32_t ans = 0U;

        for (int i = 0; i != 32; ++i, n >>= 1U)
        {
            ans <<= 1U;
            ans |= (n & 1U);
        }

        return ans;
    }

    uint32_t reverseBits2(uint32_t n) 
    {
        n = ((n & 0xaaaaaaaaU) >> 1U) | ((n & 0x55555555U) << 1U);
        n = ((n & 0xccccccccU) >> 2U) | ((n & 0x33333333U) << 2U);
        n = ((n & 0xf0f0f0f0U) >> 4U) | ((n & 0x0f0f0f0fU) << 4U);
        n = ((n & 0xff00ff00U) >> 8U) | ((n & 0x00ff00ffU) << 8U);
        n = (n >> 16U) | (n << 16U);
        return n;
    }
};