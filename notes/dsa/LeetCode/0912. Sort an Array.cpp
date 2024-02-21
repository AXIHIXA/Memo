class Solution 
{
public:
    std::vector<int> sortArray(std::vector<int> & nums) 
    {
        radixSort(nums.data(), 0, nums.size());
        // mergeSort(nums.data(), 0, nums.size());
        // quickSort(nums.data(), 0, nums.size() - 1);
        return nums;
    }

private:
    // a[lo, hi)
    void radixSort(int * a, int lo, int hi) 
    {
        // Offset indices into a[0, hi). 
        a += lo;
        hi -= lo;
        
        // Offset values into non-negative, radix sort, then offset back. 
        int minimum = *std::min_element(a, a + hi);
        for (int i = 0; i < hi; ++i) a[i] -= minimum;
        radixSortImpl(a, 0, hi);
        for (int i = 0; i < hi; ++i) a[i] += minimum;
    }

    // a[lo, hi), MUST be all non-negative. 
    void radixSortImpl(int * a, int lo, int hi, int base = 10)
    {
        cnt.resize(base);
        
        // Offset into a[0, hi). 
        a += lo;
        hi -= lo;
        tmp.resize(hi);

        // Number of bits in radix base. 
        int bits = 0;
        for (int x = *std::max_element(a, a + hi); 0 < x; x /= base) ++bits;
        
        for (int offset = 1; 0 < bits; offset *= base, --bits)
        {
            // Count bits into culmulative sum. 
            // Block write-back in REVERSE order for stability. 
            std::fill_n(cnt.data(), base, 0);
            for (int i = 0; i < hi; ++i) ++cnt[(a[i] / offset) % base];
            for (int i = 1; i < base; ++i) cnt[i] += cnt[i - 1];
            for (int i = hi - 1; 0 <= i; --i) tmp[--cnt[(a[i] / offset) % base]] = a[i];
            std::copy_n(tmp.data(), hi, a);
        }
    }

private:
    // a[lo, hi)
    void mergeSort(int * a, int lo, int hi)
    {
        if (hi < lo + 2) return;
        
        // Offset into a[0, hi). 
        a += lo;
        hi -= lo;
        tmp.resize(hi);
        
        for (int size = 1; size < hi; size <<= 1)
        {
            for (int i = 0, j, k; i < hi; i += (size << 1))
            {
                j = i + size;
                if (hi <= j) break;
                k = std::min(j + size, hi);
                merge(a, i, j, k);
            }
        }
    }

    // a[lo, hi)
    void merge(int * a, int lo, int mi, int hi)
    {
        std::copy(a + lo, a + mi, tmp.data() + lo);
        int * b = tmp.data() + lo;
        const int m = mi - lo;

        int * c = a + mi;
        const int n = hi - mi;

        for (int i = lo, j = 0, k = 0; j < m || k < n; )
        {
            if (j < m && (n <= k || b[j] <= c[k])) a[i++] = b[j++];
            if (k < n && (m <= j || c[k] <  b[j])) a[i++] = c[k++];
        }
    }

private:
    // a[lo, hi]
    void quickSort(int * a, int lo, int hi)
    {
        if (hi < lo + 1) return;

        std::stack<std::pair<int, int>> st;
        st.emplace(lo, hi);

        while (!st.empty())
        {
            std::tie(lo, hi) = st.top();
            st.pop();
            auto [ll, rr] = partition(a, lo, hi);
            if (rr + 1 < hi) st.emplace(rr + 1, hi);
            if (lo < ll - 1) st.emplace(lo, ll - 1);
        }
    }

    // a[lo, hi]
    std::pair<int, int> partition(int * a, int lo, int hi)
    {
        int p = a[lo + std::rand() % (hi - lo + 1)];
        int mi = lo;

        while (mi <= hi)
        {
            if (a[mi] < p) std::swap(a[lo++], a[mi++]);
            else if (a[mi] == p) ++mi;
            else std::swap(a[hi--], a[mi]);
        }

        return {lo, hi};
    }

private:
    // Helper space used for merge sort and radix sort routines. 
    std::vector<int> cnt;
    std::vector<int> tmp;
};

