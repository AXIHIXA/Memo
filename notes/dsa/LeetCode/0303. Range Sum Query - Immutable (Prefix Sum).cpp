class NumArray
{
public:
    NumArray(std::vector<int> & nums) : ps(nums.size() + 1, 0)
    {
        std::inclusive_scan(nums.cbegin(), nums.cend(), ps.begin() + 1, std::plus<>(), 0);
    }
    
    int sumRange(int left, int right)
    {
        return ps[right + 1] - ps[left];
    }

private:
    std::vector<int> ps;
};

/**
 * Your NumArray object will be instantiated and called as such:
 * NumArray* obj = new NumArray(nums);
 * int param_1 = obj->sumRange(left,right);
 */