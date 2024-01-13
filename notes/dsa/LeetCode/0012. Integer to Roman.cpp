class Solution 
{
public:
    string intToRoman(int num) 
    {
        string ans;

        for (int i = 0; i != kN; ++i)
        {
            while (kV[i] <= num)
            {
                ans += s[i];
                num -= kV[i];
            }
        }

        return ans;
    }

private:
    static constexpr int kN {13};
    static constexpr int kV[] {1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1};
    static string s[];
};

string Solution::s[] {"M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"};