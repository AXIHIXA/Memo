class Solution 
{
public:
    int findKthLargest(std::vector<int> & nums, int k) 
    {
        // k-th largest (1-indexed) == (size - k)-th smallest (0 indexed). 
        return quickSelect(nums.data(), nums.size(), nums.size() - k);
    }

private:
    // a[lo, hi]. 
    static std::pair<int, int> partition(int * a, int lo, int hi)
    {
        int pivot = a[lo + (std::rand() % (hi - lo + 1))];
        int mi = lo;

        while (mi <= hi)
        {
            if (a[mi] < pivot)       std::swap(a[lo++], a[mi++]);
            else if (a[mi] == pivot) ++mi;
            else                     std::swap(a[hi--], a[mi]); 
        }

        return {lo, hi};
    }

    // a[0, len), k-th smallest (0-indexed). 
    static int quickSelect(int * a, int len, int k)
    {
        int lo = 0, hi = len;

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