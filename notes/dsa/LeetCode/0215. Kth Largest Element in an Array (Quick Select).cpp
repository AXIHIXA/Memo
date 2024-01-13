class Solution 
{
public:
    int findKthLargest(vector<int> & nums, int k) 
    {
        return quickSelect(nums, k - 1);
    }

private:
    int quickSelect(std::vector<int> & a, int k)
    {
        for (int lo = 0, hi = static_cast<int>(a.size()) - 1; lo < hi; )
        {
            int i = lo;
            int j = hi;
            int pivot = a[lo];

            while (i < j)
            {
                // ">" for largest and "<" for smallest
                while (i < j and pivot > a[j])  --j;
                if (i < j)  a[i++] = a[j];
                while (i < j and a[i] > pivot)  ++i;
                if (i < j)  a[j--] = a[i];
            }

            a[i] = pivot;

            if (k <= i)  hi = i - 1;
            if (i <= k)  lo = i + 1;
        }

        return a[k];
    }
};