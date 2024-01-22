class Solution 
{
public:
    std::string minWindow(std::string s, std::string t)
    {
        std::vector<int> count(128, 0);
        for (char c : t) ++count[c];
        
        int bestLeft = -1;
        int minLength = s.length() + 1;

        for (int l = 0, r = 0, required = t.length(); r < s.length(); ++r) 
        {
            if (0 <= --count[s[r]]) --required;
            
            while (required == 0) 
            {
                if (r - l + 1 < minLength) 
                {
                    bestLeft = l;
                    minLength = r - l + 1;
                }

                if (0 < ++count[s[l++]]) ++required;
            }
        }

        return bestLeft == -1 ? "" : s.substr(bestLeft, minLength);
    }

private:
    static std::string minWindowHashMap(std::string s, std::string t) 
    {
        int m = s.size(), n = t.size();
        if (m < n) return "";

        std::unordered_map<char, int> tIdx;

        for (int i = 0; i != n; ++i)
        {
            if (auto it = tIdx.find(t[i]); it != tIdx.end())
            {
                ++it->second;
            }
            else
            {
                tIdx.emplace_hint(it, t[i], 1);
            }
        }

        std::vector<std::pair<char, int>> ss;

        for (int i = 0; i != m; ++i)
        {
            if (tIdx.find(s[i]) != tIdx.end())
            {
                ss.emplace_back(s[i], i);
            }
        }

        std::unordered_map<char, int> windowCount;
        for (auto [c, i] : tIdx) windowCount.emplace(c, 0);
        int l = -1, r = -1;

        for (int ll = 0, rr = 0, rrMax = ss.size(), charsFound = 0; rr < rrMax; ++rr)
        {
            if (++windowCount.at(ss[rr].first) <= tIdx.at(ss[rr].first))
            {
                ++charsFound;
            }

            // Try and contract the window till the point where it ceases to be 'desirable'.
            while (ll <= rr && charsFound == n)
            {
                if (l == -1 || ss[rr].second - ss[ll].second < r - l)
                {
                    l = ss[ll].second;
                    r = ss[rr].second;
                }

                if (--windowCount.at(ss[ll].first) < tIdx.at(ss[ll].first))
                {
                    --charsFound;
                }

                ++ll;
            }
        }

        return l == -1 ? "" :  s.substr(l, r - l + 1);
    }
};