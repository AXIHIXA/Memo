int init = []
{
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);
    std::cout.tie(nullptr);
    return 0;
}();

class Solution 
{
public:
    int romanToInt(string s) 
    {   
        int ans = 0;
        
        for (int i = 0; i < s.size(); )
        {
            for (int j = 0; j != kN; ++j)
            {
                if (int len = sy[j].size(); s.substr(i, len) == sy[j])
                {
                    ans += kV[j];
                    i += len;
                    break;
                }
            }
        }
        
        return ans;
    }

private:
    static constexpr int kN {13};
    static constexpr int kV[] {1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1};
    static string sy[];
};

string Solution::sy[] {"M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"};
