static const int init = []
{
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);
    std::cout.tie(nullptr);
    return 0;
}();

class Solution
{
public:
    int maximumXOR(std::vector<int> & nums)
    {
        // An operation could unset one or more bits of a number. 
        // However it could NOT set a zero bit. 
        // Thus, the xor result could have a one bit
        // as long as any x in nums has this bit set. 
        int ans = 0;
        for (int x : nums) ans |= x;
        return ans;
    }
};