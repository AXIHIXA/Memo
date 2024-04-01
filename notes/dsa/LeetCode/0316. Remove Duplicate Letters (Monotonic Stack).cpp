class Solution
{
public:
    std::string removeDuplicateLetters(std::string s)
    {
        std::array<int, 26> freq = {0};

        for (char c : s)
        {
            ++freq[c - 'a'];
        }

        std::array<bool, 26> contains = {false};
        std::string stk;
        stk.reserve(s.size());

        for (char c : s)
        {
            if (!contains[c - 'a'])
            {
                while (!stk.empty() && c < stk.back() && 0 < freq[stk.back() - 'a'])
                {
                    contains[stk.back() - 'a'] = false;
                    stk.pop_back();
                }
                
                contains[c - 'a'] = true;
                stk += c;
            }

            --freq[c - 'a'];
        }

        return stk;
    }
};