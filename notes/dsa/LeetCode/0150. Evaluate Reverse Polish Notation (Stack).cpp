class Solution
{
public:
    int evalRPN(std::vector<std::string> & tokens)
    {
        if (tokens.size() == 1) return std::stoi(tokens.front());
        
        std::stack<int> st;
        int l, r;

        for (const auto & s : tokens)
        {
            if (std::isdigit(s[0]) || (1 < s.size() && (s[0] == '-' || s[0] == '+') && std::isdigit(s[1])))
            {
                st.emplace(std::stoi(s));
                continue;
            }

            r = st.top();
            st.pop();
            l = st.top();
            st.pop();

            switch (s[0])
            {
                case '+':
                    st.emplace(l + r);
                    break;
                case '-':
                    st.emplace(l - r);
                    break;
                case '*':
                    st.emplace(l * r);
                    break;
                case '/':
                    st.emplace(l / r);
                    break;
            }
        }

        return st.top();
    }
};