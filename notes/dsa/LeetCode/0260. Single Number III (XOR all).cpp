static const int init = []
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
    std::vector<int> singleNumber(std::vector<int> & nums)
    {
        // Suppose these two numbers are a and b, xorAll == a ^ b. 
        int a = 0;
        int b = 0;
        long long aXorB = 0;
        for (int x : nums) aXorB ^= x;

        // a and b differs at bit mask, 
        // all numbers could be dividied into 2 groups, 
        // each containing either a or b but not both. 
        long long mask = aXorB & (-aXorB);

        for (int x : nums)
        {
            if (x & mask) a ^= x;
            else b ^= x;
        }

        return {a, b};
    }
};