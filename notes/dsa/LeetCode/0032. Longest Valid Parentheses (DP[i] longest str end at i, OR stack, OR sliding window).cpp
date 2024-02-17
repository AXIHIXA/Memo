class Solution
{
public:
    int longestValidParentheses(std::string s)
    {
        auto n = static_cast<const int>(s.size());
        int ans = 0;
        int ll = 0;
        int rr = 0;

        for (int i = 0; i < n; ++i)
        {
            if (s[i] == '(') ++ll;
            else ++rr;

            if (ll == rr) ans = std::max(ans, rr << 1);
            else if (ll < rr) ll = rr = 0;
        }

        ll = rr = 0;

        for (int i = n - 1; 0 <= i; --i)
        {
            if (s[i] == '(') ++ll;
            else ++rr;

            if (ll == rr) ans = std::max(ans, ll << 1);
            else if (rr < ll) ll = rr = 0;
        }

        return ans;
    }

private:
    static int longestValidParenthesesStack(std::string s)
    {
        auto n = static_cast<const int>(s.size());
        int ans = 0;
        std::stack<int> st;
        st.emplace(-1);

        for (int i = 0; i < n; ++i)
        {
            if (s[i] == '(')
            {
                st.emplace(i);
            }
            else
            {
                st.pop();

                if (st.empty()) st.emplace(i);
                else ans = std::max(ans, i - st.top());
            }
        }

        return ans;
    }

    static int longestValidParenthesesDp(std::string s)
    {
        s = ")" + s;
        auto n = static_cast<const int>(s.size());

        // dp[i]: Longest valid substr ending at s[i]. 
        std::vector<int> dp(n, 0);

        for (int i = 1; i < n; ++i)
        {
            if (s[i] == ')' && s[i - dp[i - 1] - 1] == '(')
            {
                dp[i] = dp[i - 1] + dp[i - dp[i - 1] - 2] + 2;
            }
        }

        return *std::max_element(dp.cbegin(), dp.cend());
    }
};