class Solution
{
public:
    int hammingWeight(uint32_t n)
    {
        n = ((n & 0xaaaaaaaaU) >> 1U) + (n & 0x55555555U);
        n = ((n & 0xccccccccU) >> 2U) + (n & 0x33333333U);
        n = ((n & 0xf0f0f0f0U) >> 4U) + (n & 0x0f0f0f0fU);
        n = ((n & 0xff00ff00U) >> 8U) + (n & 0x00ff00ffU);
        n = ((n & 0xffff0000U) >> 16U) + (n & 0x0000ffffU);
        return n;
    }
};