class Solution 
{
public:
    int removeElement(vector<int> & nums, int val) 
    {
        int i = 0;

        for (int e : nums)
        {
            if (e != val)
            {
                nums[i++] = e;
            }
        }

        return i;
    }
};