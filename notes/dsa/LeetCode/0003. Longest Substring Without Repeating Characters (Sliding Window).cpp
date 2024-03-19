class Solution
{
public:
    int lengthOfLongestSubstring(std::string s)
    {
        auto n = static_cast<const int>(s.size());

        std::array<int, 256> leftmost;
        std::fill(leftmost.begin(), leftmost.end(), -1);

        int ans = 0;

        for (int ll = 0, rr = 0; rr < n; ++rr)
        {
            if (leftmost[s[rr]] == -1)
            {
                ans = std::max(ans, rr - ll + 1);
            }
            else
            {
                while (ll <= leftmost[s[rr]])
                {
                    leftmost[s[ll]] = -1;
                    ++ll;
                }
            }

            leftmost[s[rr]] = rr;

            // std::cout << s.substr(ll, rr - ll + 1) << '\n';
        }

        return ans;
    }
};