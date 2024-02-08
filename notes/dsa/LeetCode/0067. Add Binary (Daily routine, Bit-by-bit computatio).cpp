class Solution
{
public:
    std::string addBinary(std::string a, std::string b)
    {
        auto m = static_cast<int>(a.size());
        auto n = static_cast<int>(b.size());
        
        std::string ans;
        ans.reserve(1 + std::max(m, n));

        for (int i = m - 1, j = n - 1, carry = 0; 0 <= i || 0 <= j || carry; )
        {
            if (0 <= i) carry += a[i--] - '0';
            if (0 <= j) carry += b[j--] - '0';
            ans += (carry & 1) + '0';
            carry >>= 1;
        }

        std::reverse(ans.begin(), ans.end());
        return ans;
    }
};