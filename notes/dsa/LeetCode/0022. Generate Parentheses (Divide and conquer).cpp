class Solution
{
public:
    std::vector<std::string> generateParenthesis(int n)
    {
        if (n == 0) return {""};
        
        std::vector<std::string> ans;

        // A valid str starts with '(', 
        // we loop location of its pairing ')', 
        // F(N) = ( F(L) ) F(R). 
        // Guaranteed no duplicates. 
        for (int left = 0; left < n; ++left)
        {
            for (const std::string & l : generateParenthesis(left))
            {
                for (const std::string & r : generateParenthesis(n - 1 - left))
                {
                    ans.emplace_back("(" + l + ")" + r);
                }
            }
        }

        return ans;
    }
};
