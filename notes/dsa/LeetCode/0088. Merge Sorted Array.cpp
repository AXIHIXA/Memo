class Solution 
{
public:
    void merge(vector<int>& nums1, int m, vector<int>& nums2, int n) 
    {
        vector<int> tmp;
        tmp.reserve(m);
        tmp.insert(tmp.end(), nums1.cbegin(), nums1.cend());
        
        for (int i = 0, j = 0, k = 0; j < m || k < n; )
        {
            if ((j < m) && (n <= k || tmp[j] < nums2[k]))
            {
                nums1[i++] = tmp[j++];
            }
            
            if ((k < n) && (m <= j || nums2[k] <= tmp[j]))
            {
                nums1[i++] = nums2[k++];
            }

        }
    }
};