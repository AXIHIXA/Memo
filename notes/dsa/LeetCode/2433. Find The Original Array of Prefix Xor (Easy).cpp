class Solution 
{
public:
    vector<int> findArray(vector<int> & pref) 
    {
        std::vector<int> arr;
        arr.reserve(pref.size());

        arr.emplace_back(pref[0]);

        for (int i = 1; i < pref.size(); ++i)
        {
            arr.emplace_back(pref[i] ^ pref[i - 1]);
        }

        return arr;
    }
};