class Solution
{
public:
    std::string minWindow(std::string s, std::string t)
    {
        auto m = static_cast<const int>(s.size());
        auto n = static_cast<const int>(t.size());

        std::array<int, 256> needs = {0};
        for (char c : t)

        {
            ++needs[c];
        }

        int charsNeeded = n;

        int ansLl = -1;
        int ansRr = m + 1;

        for (int ll = 0, rr = 0; rr < m; ++rr)
        {
            if (0 < needs[s[rr]])
            {
                --charsNeeded;
            }

            --needs[s[rr]];
            
            while (ll <= rr && charsNeeded == 0)
            {
                if (rr - ll < ansRr - ansLl)
                {
                    ansLl = ll;
                    ansRr = rr;
                }

                if (needs[s[ll]] == 0)
                {
                    ++charsNeeded;
                }

                ++needs[s[ll]];
                ++ll;
            }
        }

        return ansLl == -1 ? "" : s.substr(ansLl, ansRr - ansLl + 1);
    }
};