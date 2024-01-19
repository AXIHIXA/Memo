class Solution 
{
public:
    bool isPalindrome(string s) 
    {
        if (s.empty()) return true;

        for (int ll = 0, rr = s.size() - 1; ll < rr; ++ll, --rr)
        {
            while (ll < rr && !std::isalnum(s[ll])) ++ll;
            while (ll < rr && !std::isalnum(s[rr])) --rr;
            if (std::tolower(s[ll]) != std::tolower(s[rr])) return false;
        }

        return true;
    }
};