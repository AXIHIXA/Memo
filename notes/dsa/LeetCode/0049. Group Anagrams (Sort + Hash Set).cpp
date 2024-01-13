class Solution 
{
public:
    vector<vector<string>> groupAnagrams(vector<string> & strs)
    {
        vector<vector<string>> ans;
        unordered_map<string, int> index;

        for (const string & str : strs)
        {
            string k = str;
            sort(k.begin(), k.end());
            
            if (auto it = index.find(k); it == index.end())
            {
                index[k] = ans.size();
                ans.push_back({});
                ans.back().emplace_back(str);
            }
            else
            {
                ans[it->second].emplace_back(str);
            }
        }

        return ans;
    }
};
