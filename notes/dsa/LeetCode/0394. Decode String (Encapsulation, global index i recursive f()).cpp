class Solution
{
public:
    std::string decodeString(std::string s)
    {
        i = 0;
        return f(s);
    }

private:
    std::string f(const std::string & s)
    {
        std::string ans;

        int cur = 0;

        for (; i < s.size() && s[i] != ']'; ++i)
        {
            char c = s[i];

            if (std::isdigit(c))
            {
                cur = cur * 10 + c - '0';
            }
            else if (c != '[')
            {
                ans += c;
            }
            else
            {
                ++i;
                std::string bkt = f(s);

                for (int _ = 0; _ < cur; ++_)
                {
                    ans += bkt;
                }

                cur = 0;
            }
        }

        return ans;
    }

    int i = 0;
};