class Solution
{
public:
    int calculate(std::string s)
    {
        i = 0;
        return f(s);
    }

private:
    int f(const std::string & s)
    {
        auto n = static_cast<const int>(s.size());

        std::vector<char> oper;
        std::vector<int> oprd;

        int cur = 0;

        for ( ; i < n && s[i] != ')'; ++i)
        {
            int c = s[i];

            if (std::isdigit(c))
            {
                cur = cur * 10 + c - '0';
            }
            else if (c != '(')  // + - * /
            {
                // Note: ')' will be consumed automatically
                // by the for update after recursive f() call. 
                push(oper, oprd, c, cur);
                cur = 0;
            }
            else  // (
            {
                ++i;
                cur = f(s);
            }
        }

        // Consume the last operand. 
        // The operator could be anything. 
        push(oper, oprd, '+', cur);

        return compute(oper, oprd);
    }

    void push(std::vector<char> & oper, std::vector<int> & oprd, char op, int cur)
    {
        if (oprd.empty() || oper.back() == '+' || oper.back() == '-')
        {
            oper.emplace_back(op);
            oprd.emplace_back(cur);
        }
        else
        {
            oprd.back() = oper.back() == '*' ? oprd.back() * cur : oprd.back() / cur;
            oper.back() = op;
        }
    }

    int compute(std::vector<char> & oper, std::vector<int> & oprd)
    {
        auto n = static_cast<const int>(oprd.size());
        int ans = oprd[0];

        for (int i = 1; i < n; ++i)
        {
            ans = oper[i - 1] == '+' ? ans + oprd[i] : ans - oprd[i];
        }

        return ans;
    }

    // Current index-of-processing of string s. 
    int i = 0;
};