class Solution
{
public:
    int distinctSubseqII(std::string s)
    {
        // dp[i]: Number of unique subseqs ending with 'a' + i. 
        std::vector<int> dp(26, 0);

        // Number of all unique subseqs (including empty str, remove when returning).
        int ans = 1;

        for (char c : s)
        {
            // Number of unique subseqs ending at this char c, 
            // without duplicating previously-discovered subseqs ending with char c. 
            int newAdd = (ans - dp[c - 'a'] + p) % p;

            // Number of unique subseqs ending with char c. 
            dp[c - 'a'] = (dp[c - 'a'] + newAdd) % p;

            // Number of all known unique subseqs. 
            ans = (ans + newAdd) % p;
        }

        return (ans - 1 + p) % p;
    }

private:
    static constexpr int p = 1'000'000'007;
};