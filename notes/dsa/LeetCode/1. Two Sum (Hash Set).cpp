class Solution 
{
public:
    vector<int> twoSum(vector<int> & nums, int target) 
    {
        std::unordered_map<int, int> dic;

        for (int i = 0; i != nums.size(); ++i)
        {
            dic.insert({nums[i], i});
            
            if (auto it = dic.find(target - nums[i]); it != dic.end() and it->second != i)
            {
                return {i, it->second};
            }
        }

        return {};
    }
};