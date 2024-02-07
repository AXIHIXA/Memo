class Solution 
{
public:
    vector<int> twoSum(vector<int> & numbers, int target) 
    {
        int lo = 0, hi = numbers.size() - 1, sum;

        while (lo < hi)
        {
            sum = numbers[lo] + numbers[hi];

            if (sum == target)
            {
                return {lo + 1, hi + 1};
            }
            else if (sum < target)
            {
                ++lo;
            }
            else
            {
                --hi;
            }
        }

        return {-1, -1};
    }
};