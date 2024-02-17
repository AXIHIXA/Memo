class Solution
{
public:
    int strangePrinter(std::string s)
    {
        auto n = static_cast<const int>(s.size());

        // O(n**3) DP：遍历子区间长度、左端点以及分割点。

        // 引理:
        // 必然存在至少一个最优解，满足每一次打印都最终确定打印区间最左侧的字符，
        // 即最左侧的字符不会被之后的打印步骤覆盖。
        // 证明：
        // 首先，一个最优解的每一步都要确定至少一个字符，不然这次打印可以直接省略而不影响最终结果。
        // 假设某最优解的某一步确定的不是最左侧的字符。
        // 如果把这一步的打印范围调整为“从它确定的那个字符开始”，最终结果依旧不变。
        // 那么这个最优解可以被调整为满足引理要求的样子。

        // 考虑“引理式”打印一个子区间s[i..j]，则必然有一次打印最终确定了s[i]。
        // 这次打印要么只打印了s[i]一个字符，要么打印了s[i..k]这个区间
        // （此时s[i] == s[k]，否则s[k]还要被别人覆盖，直接不用打印）。
        // 那么状态转移方程如下：
        // dp[i][j]表示打印s[i..j]所需步骤数，
        // dp[i][j] = max(1 + dp[i + 1][j], dp[i][k - 1] + dp[k + 1][j] for s[k] == s[j])。
        // 其中第二项，dp[i][k - 1]是因为打印左侧时直接处理掉了s[k]，s[k]以后不用管了。

        std::vector<std::vector<int>> dp(n + 1, std::vector<int>(n + 1, 0));

        for (int len = 1; len <= n; ++len)
        {
            for (int i = 0, j; i + len - 1 < n; ++i)
            {
                j = i + len - 1;
                dp[i][j] = 1 + dp[i + 1][j];

                for (int k = i + 1; k <= j; ++k)
                {
                    if (s[i] == s[k])
                    {
                        dp[i][j] = std::min(dp[i][j], dp[i][k - 1] + dp[k + 1][j]);
                    }
                }
            }
        }

        return dp[0][n - 1];
    }
};
