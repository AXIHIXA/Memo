class Solution 
{
public:
    vector<vector<int>> fourSum(vector<int>& nums, int target) 
    {
        long long t0 = target;
        
        const int kN = nums.size();
        sort(nums.begin(), nums.end());

        vector<vector<int>> ans;

        for (int i = 0; i < kN - 3; ++i)
        {
            for (int j = i + 1; j < kN - 2; ++j)
            {
                long long t1 = t0 - nums[i] - nums[j];

                int lo = j + 1, hi = kN - 1;

                while (lo < hi)
                {
                    long long su = static_cast<long long>(nums[lo]) + 
                                   static_cast<long long>(nums[hi]);
                    
                    if (su < t1)
                    {
                        ++lo;
                    }
                    else if (t1 < su)
                    {
                        --hi;
                    }
                    else
                    {
                        ans.push_back({nums[i], nums[j], nums[lo], nums[hi]});
                        int ll = lo, rr = hi;
                        while (lo < hi and nums[lo] == nums[ll]) ++lo;
                        while (lo < hi and nums[hi] == nums[rr]) --hi;
                    }

                    while (j + 1 < kN and nums[j] == nums[j + 1]) ++j;
                }
            }

            while (i + 1 < kN and nums[i] == nums[i + 1]) ++i;
        }

        return ans;
    }
};