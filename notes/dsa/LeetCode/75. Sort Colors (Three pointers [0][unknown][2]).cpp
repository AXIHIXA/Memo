class Solution 
{
public:
    void sortColors(vector<int> & nums) 
    {
        int left = 0, mid = 0, right = nums.size() - 1;

        while (mid <= right)
        {
            switch (nums[mid])
            {
            case 0:
                swap(nums[left++], nums[mid++]);
                break;
            case 1:
                ++mid;
                break;
            default:
                swap(nums[right--], nums[mid]);
                break;
            }
        }
    }
};