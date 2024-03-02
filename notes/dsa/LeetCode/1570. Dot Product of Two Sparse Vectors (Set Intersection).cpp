class SparseVector
{
public:
    SparseVector(vector<int> & nums)
    {
        auto n = static_cast<const int>(nums.size());

        for (int i = 0; i < n; ++i)
        {
            if (nums[i] != 0)
            {
                index.emplace_back(i);
                row.emplace_back(nums[i]);
            }
        }
    }
    
    // Return the dotProduct of two sparse vectors
    int dotProduct(SparseVector & vec)
    {
        auto m = static_cast<const int>(index.size());
        auto n = static_cast<const int>(vec.index.size());

        int ans = 0;

        for (int i = 0, j = 0; i < m && j < n; )
        {
            if (index[i] == vec.index[j])
            {
                ans += row[i] * vec.row[j];
                ++i;
                ++j;
            }
            else if (index[i] < vec.index[j])
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

private:
    std::vector<int> index;
    std::vector<int> row;
};

// Your SparseVector object will be instantiated and called as such:
// SparseVector v1(nums1);
// SparseVector v2(nums2);
// int ans = v1.dotProduct(v2);