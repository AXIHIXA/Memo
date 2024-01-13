class Solution 
{
public:
    int findShortestSubArray(vector<int> & nums) 
    {
        unordered_map<int, int> count, first;
        int res = 0, degree = 0;

        for (int i = 0; i < nums.size(); ++i)
        {
            if (first.count(nums[i]) == 0)
            {
                first[nums[i]] = i;
            }

            ++count[nums[i]];

            if (degree < count[nums[i]]) 
            {
                degree = count[nums[i]];
                res = i - first[nums[i]] + 1;
            } 
            else if (degree == count[nums[i]])
            {
                res = min(res, i - first[nums[i]] + 1);
            }
                
        }

        return res;
    }
};