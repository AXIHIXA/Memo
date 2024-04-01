class Solution
{
public:
    std::string removeDuplicateLetters(std::string s)
    {
        std::array<int, 26> freq = {0};
        std::array<bool, 26> contains = {false};

        for (char c : s)
        {
            ++freq[c - 'a'];
        }

        auto n = static_cast<const int>(s.size());
        std::string stk;
        stk.reserve(n);

        for (char c : s)
        {
            if (!contains[c - 'a'])
            {
                while (!stk.empty() && c < stk.back() && 0 < freq[stk.back() - 'a'])
                {
                    contains[stk.back() - 'a'] = false;
                    stk.pop_back();
                }

                stk += c;
                contains[c - 'a'] = true;
            }

            --freq[c - 'a'];
        }

        return stk;
    }
};