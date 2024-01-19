class Solution
{
public:
    // Hash Map. 
    // Works for multiple queries on single target. 
    bool isSubsequence(std::string s, std::string t)
    {
        std::unordered_map<char, std::vector<int>> idx;

        int m = t.size();

        for (int i = 0; i != m; ++i)
        {
            if (auto it = idx.find(t[i]); it != idx.end())
            {
                it->second.emplace_back(i);
            }
            else
            {
                idx.insert(it, {t[i], {i}});
            }
        }

        for (int j = 0, n = s.size(), currentMatchIndex = -1; j != n; ++j)
        {
            auto it = idx.find(s[j]);
            if (idx.find(s[j]) == idx.end()) return false;

            bool match = false;

            for (int i : it->second)
            {
                if (currentMatchIndex < i)
                {
                    currentMatchIndex = i;
                    match = true;
                    break;
                }
            }

            if (!match) return false;
        }

        return true;
    }

private:
    // Two Pointers Greedy Match. 
    // Works for single query. 
    bool isSubsequenceTwoPointers(std::string s, std::string t)
    {
        int i = 0, m = t.size();
        int j = 0, n = s.size();

        while (i < m && j < n)
        {
            t[i] == s[j] ? ++i, ++j : ++i;
        }

        return j == n;
    }
};