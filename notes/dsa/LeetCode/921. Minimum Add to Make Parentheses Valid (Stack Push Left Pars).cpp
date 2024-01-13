class Solution 
{
public:
    int minAddToMakeValid(string s) 
    {
        stack<char> st;
        int cnt = 0;

        for (char c : s)
        {
            if (c == '(')
            {
                st.push(c);
            }
            else if (c == ')')
            {
                st.empty() ? static_cast<void>(++cnt) : st.pop();
            }
        }

        return st.size() + cnt;
    }
};