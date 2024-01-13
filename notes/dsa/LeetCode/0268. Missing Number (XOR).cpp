class Solution 
{
public:
    int missingNumber(vector<int> & nums) 
    {
        auto len = static_cast<int>(nums.size());
        int missing = len;

        for (int i = 0; i != len; ++i)
        {
            missing ^= (i ^ nums[i]);
        }

        return missing;
    }
};