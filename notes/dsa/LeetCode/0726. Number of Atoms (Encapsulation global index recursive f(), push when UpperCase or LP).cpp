class Solution
{
public:
    std::string countOfAtoms(std::string formula)
    {
        i = 0;
        std::map<std::string, int> frequency = f(formula);
        std::string ans;

        for (const auto & [elem, freq] : frequency)
        {
            ans += elem;
            if (1 < freq) ans += std::to_string(freq);
        }

        return ans;
    }

private:
    std::map<std::string, int> f(const std::string & s)
    {
        int curr = 0;
        std::string elem;
        std::map<std::string, int> prev;

        std::map<std::string, int> ans;
        
        for ( ; i < s.size() && s[i] != ')'; ++i)
        {
            char c = s[i];
            
            if (std::isdigit(c))
            {
                curr = curr * 10 + c - '0';
            }
            else if (c != '(')
            {
                if (std::islower(c))
                {
                    elem += c;
                }
                else
                {
                    push(ans, prev, elem, curr);
                    curr = 0;
                    elem = c;
                }
            }
            else
            {
                push(ans, prev, elem, curr);
                ++i;
                prev = f(s);
            }
        }

        push(ans, prev, elem, curr);

        return ans;
    }

    void push(
            std::map<std::string, int> & ans, 
            std::map<std::string, int> & prev, 
            std::string & elem, 
            int & curr)
    {
        if (!elem.empty())
        {
            ans[elem] += std::max(1, curr);
            elem.clear();
        }

        if (!prev.empty())
        {
            for (const auto & [k, v] : prev)
            {
                ans[k] += std::max(1, curr) * v;
            }

            prev.clear();
        }

        curr = 0;
    }

    int i = 0;
};