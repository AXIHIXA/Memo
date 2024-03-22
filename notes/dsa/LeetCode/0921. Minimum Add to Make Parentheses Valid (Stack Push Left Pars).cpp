class Solution
{
public:
    int minAddToMakeValid(std::string s)
    {
        int ans = 0;
        int leftParentheses = 0;

        for (char c : s)
        {
            if (c == '(')
            {
                ++leftParentheses;
            }
            else
            {
                if (!leftParentheses)
                {
                    ++ans;
                }
                else
                {
                    --leftParentheses;
                }
            }
        }

        return ans + leftParentheses;
    }
};