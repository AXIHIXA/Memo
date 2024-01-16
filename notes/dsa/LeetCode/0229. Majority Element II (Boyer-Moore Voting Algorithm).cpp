class Solution 
{
public:
    vector<int> majorityElement(vector<int> & nums)
    {
        if (nums.size() == 1) return nums;
        
        // At most two elements whose frequency > n / 3. 
        // Boyer-Moore Voting Algorithm. 
        int a1 = nums[0], a2 = nums[1];
        int count1 = 0, count2 = 0;

        for (int e : nums)
        {
            if (a1 == e)
            {
                ++count1;
            }
            else if (a2 == e)
            {
                ++count2;
            }
            else if (count1 == 0)
            {
                a1 = e;
                ++count1;
            }
            else if (count2 == 0)
            {
                a2 = e;
                ++count2;
            }
            else
            {
                --count1;
                --count2;
            }
        }

        std::vector<int> ret;

        count1 = 0, count2 = 0;

        for (int e : nums)
        {
            if (a1 == e) ++count1;
            else if (a2 == e) ++count2;  // Note this else; in case a1 == a2!
        }

        int n3 = nums.size() / 3;

        if (n3 < count1) ret.push_back(a1);
        if (n3 < count2) ret.push_back(a2);

        return ret;
    }
};