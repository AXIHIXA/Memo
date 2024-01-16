class Solution
{
public:
    // Stable unique routine for sorted array s.t. each element appears at most twice. 
    int removeDuplicates(vector<int> & nums)
    {
        int i = 0;

        for (int e : nums)
        {
            if (i < 2 || nums[i - 2] != e)
            {
                nums[i++] = e;
            }
        }

        return i;
    }
};