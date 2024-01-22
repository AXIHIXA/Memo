class Solution 
{
public:
    std::vector<std::vector<std::string>> groupAnagrams(std::vector<std::string> & strs)
    {
        if (strs.size() == 1) return {{strs[0]}};
        
        std::unordered_map<std::string, std::vector<std::string>> idx;

        for (const std::string & str : strs)
        {
            std::string k = str;
            std::sort(k.begin(), k.end());
            auto it = idx.find(k);

            if (it == idx.end()) idx.emplace_hint(it, k, std::vector {str});
            else                 it->second.push_back(str);
        }

        std::vector<std::vector<std::string>> ans;
        for (auto & [_, vec] : idx) ans.push_back(std::move(vec));
        
        return ans;
    }
};
