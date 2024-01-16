class Solution 
{
public:
    int removeDuplicates(vector<int> & nums) 
    {
        int i = 1;

        for (int j = 1; j < nums.size(); ++j)
        {
            if (nums[i - 1] != nums[j])
            {
                nums[i++] = nums[j];
            }
        }

        return i;
    }
};