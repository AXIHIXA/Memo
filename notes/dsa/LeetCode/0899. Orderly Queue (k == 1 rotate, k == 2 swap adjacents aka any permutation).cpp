class Solution
{
public:
    std::string orderlyQueue(std::string s, int k)
    {
        // Math. 
        // k == 1 allows to "circular rotate" the string. 
        // k == 2 further allows swapping any two adjacent chars, 
        //        i.e., getting any permutation of s. 

        if (1 < k)
        {
            std::string ans = s;
            std::sort(ans.begin(), ans.end());
            return ans;
        }
        
        std::string tmp = s + s;
        char * p = tmp.data();
        auto len = static_cast<const std::size_t>(s.size());

        std::string_view ans(p, len);
        
        for (int i = 1; i < len; ++i)
        {
            ans = std::min(ans, std::string_view(p + i, len));
        }

        return std::string(ans);
    }
};