class Solution
{
public:
    bool isIsomorphic(std::string s, std::string t)
    {
        // return firstOccuranceTransform(s) == firstOccuranceTransform(t);

        std::array<char, 138> smap, tmap;
        std::fill(smap.begin(), smap.end(), -1);
        std::fill(tmap.begin(), tmap.end(), -1);

        for (int i = 0; i < s.size(); ++i)
        {
            char c = s[i];

            if (smap[c] == -1)
            {
                smap[c] = t[i];
                // Ensure no two chars from s are mapped to the same char. 
                if (tmap[t[i]] != -1) return false;
                tmap[t[i]] = c;
            }

            s[i] = smap[c];
        }

        return s == t;
    }

private:
    std::vector<int> firstOccuranceTransform(const std::string & s)
    {
        // First Occurance Transformation. 
        // For each char in string, transform char to the index of its first occurance. 
        std::array<int, 138> smap;
        std::fill(smap.begin(), smap.end(), -1);

        std::vector<int> res(s.size());

        for (int i = 0; i != s.size(); ++i)
        {
            char c = s[i];
            if (smap[c] == -1) smap[c] = i;
            res[i] = smap[c];
        }

        return res;
    }
};
