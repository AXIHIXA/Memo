class NumArray
{
public:
    NumArray(std::vector<int> & nums) : n(nums.size() + 1), orig(nums)
    {
        tree.assign(n, 0);
        
        for (int i = 0; i != nums.size(); ++i)
        {
            add(i + 1, nums[i]);
        }
    }
    
    void update(int index, int val)
    {
        add(index + 1, val - orig[index]);
        orig[index] = val;
    }
    
    int sumRange(int left, int right)
    {
        return query(right + 1) - query(left);
    }

private:
    int query(int x)
    {
        int ans = 0;
        for (int i = x; i; i -= i & -i) ans += tree[i];
        return ans;
    }

    void add(int x, int u)
    {
        for (int i = x; i < n; i += i & -i) tree[i] += u;
    }
    
    // Binary Indexed Tree (树状数组). 
    // Segmented cumulative sum. 
    // tree[x] stores cumulative sum
    // from orig[x - (x & -x) + 1] to orig[x] (inclusive), 
    // where orig's indices start from 1 (NOT ZERO)!
    int n;  // tree.size() == orig.size() + 1
    std::vector<int> tree;
    std::vector<int> orig;
};

/**
 * Your NumArray object will be instantiated and called as such:
 * NumArray* obj = new NumArray(nums);
 * obj->update(index,val);
 * int param_2 = obj->sumRange(left,right);
 */