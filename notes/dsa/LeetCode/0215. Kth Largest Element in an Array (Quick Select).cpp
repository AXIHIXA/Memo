class Solution 
{
public:
    int findKthLargest(vector<int> & nums, int k) 
    {
        return quickSelect(nums.data(), 0, nums.size(), nums.size() - k);
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

    static int quickSelect(int * a, int lo, int hi, int k)
    {
        a += lo;
        hi -= lo;
        lo = 0;

        while (lo < hi)
        {
            auto [ll, rr] = partition(a, lo, hi - 1);
            
            if (k < ll)      hi = ll;
            else if (rr < k) lo = rr + 1;
            else             return a[k];
        }

        return -1;
    }
};