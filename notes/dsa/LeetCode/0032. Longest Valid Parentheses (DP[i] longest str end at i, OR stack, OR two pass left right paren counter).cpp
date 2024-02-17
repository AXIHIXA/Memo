class Solution
{
public:
    int longestValidParentheses(std::string s)
    {
        /*
        Let i and j denote the starting and ending indices of the longest valid subsequence.
        Note that in the forward pass after (fully) processing each character, it's always the case that left >= right. (*)
        This is in particular true after processing i-1 (immediately before processing i).

        Case 1: If immediately before i left = right, then the forward pass will detect the length of the longest valid subsequence.
        (The number of '(' and ')' is the same for any valid subseq. Thus after processing j, left = right.)

        Case 2: If immediately before i left > right, the forward pass will not detect the longest valid subsequence, but we claim the backward pass will.
        Similar to observation (*) above, note that right >= left after fully processing each element in the backward pass. We detect the longest valid subsequence in the backward pass if and only if right = left after processing j+1 (immediately before processing j).
        So what if right > left (in the backward pass immediately before processing j)?
        Well, then the maximality of our subsequence from i to j would be contradicted.
        Namely, we can increase j to j' so that (j,j'] has one more ')' than '(', and decrease i to i', so that [i',i) has one more '(' than ')'.
        The resulting subsequence from i' to j' will have longer length and will still be valid (the number of '(' and ')' will be incremented by the same amount).

        Thus, either the backward pass or forward pass (possibly both) will detect the longest valid subsequence.
        */
        
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