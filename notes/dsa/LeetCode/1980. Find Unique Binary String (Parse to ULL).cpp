class Solution 
{
public:
    string findDifferentBinaryString(vector<string> & nums) 
    {
        sort(nums.begin(), nums.end());

        unsigned long long last = stoull(nums.front(), 0, 2);
        unsigned long long res = 0;

        if (0 < last)
        {
            return string(nums.size(), '0');
        }

        for (int i = 1; i != nums.size(); ++i)
        {
            if (unsigned long long curr = stoull(nums[i], 0, 2); curr != last + 1)
            {
                res = last + 1;
                break;
            }
            else
            {
                last = curr;
            }
        }
        
        if (res == 0)
        {
            res = stoull(nums.back(), 0, 2) + 1;
        }

        string tmp(nums.size(), '0');

        for (unsigned long long num = res, i = nums.size() - 1; num; num >>= 1, --i)
        {
            tmp[i] = (num & 1) + '0';
        }
        
        return tmp;
    }
};