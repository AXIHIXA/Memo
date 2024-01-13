class Solution 
{
public:
    vector<vector<int>> fourSum(vector<int>& nums, int target) 
    {
        long long t0 = target;
        
        const int kN = nums.size();
        sort(nums.begin(), nums.end());

        set<vector<int>> s;
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
                        s.insert({nums[i], nums[lo], nums[hi], nums[j]});
                        ++lo, --hi;
                    }
                }
            }
        }

        for (auto & v : s)
        {
            ans.emplace_back(move(v));
        }

        return ans;
    }
};