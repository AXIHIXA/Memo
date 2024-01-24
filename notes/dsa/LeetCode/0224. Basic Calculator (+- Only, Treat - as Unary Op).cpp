class Solution
{
public:
    int calculate(std::string s)
    {
        std::stack<int> st;
        int res = 0, sign = 1, operand = 0;

        for (char c : s)
        {
            if (std::isdigit(c))
            {
                operand = 10 * operand + (c - '0');
            }
            else if (c == '+')
            {
                res += sign * operand;
                sign = 1;
                operand = 0;
            }
            else if (c == '-')
            {
                res += sign * operand;
                sign = -1;
                operand = 0;
            }
            else if (c == '(')
            {
                st.push(res);
                st.push(sign);
                sign = 1;
                res = 0;
            }
            else if (c == ')')
            {
                res += sign * operand;
                res *= st.top();
                st.pop();
                res += st.top();
                st.pop();
                operand = 0;
            }
        }

        return res + sign * operand;
    }
};