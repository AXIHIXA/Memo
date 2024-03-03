class Solution
{
public:
    std::vector<int> intersection(std::vector<int> & nums1, std::vector<int> & nums2)
    {
        auto m = static_cast<const int>(nums1.size());
        auto n = static_cast<const int>(nums2.size());
        std::sort(nums1.begin(), nums1.end());
        std::sort(nums2.begin(), nums2.end());

        std::vector<int> ans;

        for (int i = 0, j = 0; i < m && j < n; )
        {
            if (nums1[i] == nums2[j])
            {
                // Each element in the result must be unique. 
                if (ans.empty() || ans.back() != nums1[i])
                {
                    ans.emplace_back(nums1[i]);
                }
                
                ++i;
                ++j;
            }
            else if (nums1[i] < nums2[j])
            {
                ++i;
            }
            else
            {
                ++j;
            }
        }

        return ans;
    }
};