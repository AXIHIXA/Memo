class Solution
{
public:
    std::string decodeString(std::string s)
    {
        auto n = static_cast<const int>(s.size());

        int x = 0;
        std::stack<int> cnt;

        std::string str;
        std::stack<std::string> stk;

        for (char c : s)
        {
            if (std::isdigit(c))
            {
                x = x * 10 + c - '0';
            }
            else if (c == '[')
            {
                cnt.emplace(x);
                x = 0;

                stk.emplace(str);
                str = "";
            }
            else if (c == ']')
            {
                std::string block = stk.top();
                stk.pop();

                for (int i = 0; i < cnt.top(); ++i)
                {
                    block += str;
                }
                
                str = block;
                cnt.pop();
            }
            else
            {
                str += c;
            }
        }

        return str;
    }
};