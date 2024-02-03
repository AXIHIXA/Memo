class Solution
{
public:
    std::string getPermutation(int n, int k)
    {
        if (n == 1) return "1";

        std::array<int, 9> fac {1, 1};
        for (int i = 2; i < 9; ++i) fac[i] = i * fac[i - 1];

        std::vector<int> num(9);
        std::iota(num.begin(), num.end(), 1);

        std::string ans;
        ans.reserve(9);
        
        --k; 

        for (int i = n - 1; 0 <= i; --i)
        {
            int j = k / fac[i];
            k %= fac[i];
            ans += std::to_string(num[j]);
            num.erase(num.begin() + j);
        }

        return ans;
    }
};
