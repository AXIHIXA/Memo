class Solution 
{
public:
    bool isValid(string s) 
    {
        stack<char> st;

        auto match = [](char l, char r)
        {
            return (l == '(' and r == ')') or 
                   (l == '[' and r == ']') or
                   (l == '{' and r == '}');
        };

        for (char c : s)
        {
            if (c == '(' or c == '[' or c == '{')
            {
                st.push(c);
                continue;
            }
            
            if (not st.empty() and match(st.top(), c))
            {
                st.pop();
            }
            else
            {
                return false;
            }
        }

        return st.empty();
    }
};