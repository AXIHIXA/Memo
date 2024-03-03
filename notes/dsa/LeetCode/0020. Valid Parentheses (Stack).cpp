class Solution 
{
public:
    bool isValid(std::string s) 
    {
        auto n = static_cast<const int>(s.size());
        if (n & 1) return false;
        std::stack<char> stk;

        for (int i = 0; i < n; ++i)
        {
            if (s[i] == '(' || s[i] == '[' || s[i] == '{')
            {
                stk.emplace(s[i]);
            }
            else 
            {
                if (stk.empty()) return false;
                if (s[i] == ')' && stk.top() != '(') return false;
                if (s[i] == ']' && stk.top() != '[') return false;
                if (s[i] == '}' && stk.top() != '{') return false;
                stk.pop();
            }
        }

        return stk.empty();
    }
};