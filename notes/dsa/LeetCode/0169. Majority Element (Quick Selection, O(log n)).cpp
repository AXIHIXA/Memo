class Solution 
{
public:
    int majorityElement(vector<int> & nums) 
    {
        int lo = 0, hi = nums.size() - 1, n2 = nums.size() >> 1U;

        while (lo <= hi)
        {
            auto [ll, rr] = partition(nums.data(), lo, hi);

            if (n2 < ll) hi = ll - 1;
            else if (rr < n2) lo = rr + 1;
            else return nums[ll];
        }

        return nums[0];
    }

private:
    static std::pair<int, int> partition(int * a, int lo, int hi)
    {
        int p = a[lo + std::rand() % (hi - lo + 1)];
        int mi = lo;

        while (mi <= hi)
        {
            if (a[mi] < p)       std::swap(a[lo++], a[mi++]);
            else if (a[mi] == p) ++mi;
            else                 std::swap(a[hi--], a[mi]);
        }

        return {lo, hi};
    }
};