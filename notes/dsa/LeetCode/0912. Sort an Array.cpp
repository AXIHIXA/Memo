class Solution 
{
public:
    vector<int> sortArray(vector<int> & nums) 
    {
        mergeSortIterative(nums.data(), 0, nums.size());
        return nums;
    }

private:
    // arr[lo, hi], NOTE that it's a CLOSED inverval! 
    static int partition(int * a, int lo, int hi)
    {
        std::swap(a[lo], a[lo + std::rand() % (hi - lo + 1)]);
        int p = a[lo];

        while (lo < hi)
        {
            while (lo < hi && p < a[hi]) --hi;
            if (lo < hi) a[lo++] = a[hi];
            while (lo < hi && a[lo] < p) ++lo;
            if (lo < hi) a[hi--] = a[lo];
        }

        a[lo] = p;
        return lo;
    }

    // a[lo, hi)
    static void quickSort(int * a, int lo, int hi)
    {
        if (hi < lo + 2) return;

        int mi = partition(a, lo, hi - 1);
        quickSort(a, lo, mi);
        quickSort(a, mi + 1, hi);
    }

    // a[lo, hi)
    static void quickSortIterative(int * a, int lo, int hi)
    {
        std::stack<std::pair<int, int>> st;
        st.emplace(lo, hi);

        while (!st.empty())
        {
            auto [ll, rr] = st.top();
            st.pop();
            if (rr < ll + 2) continue;

            int mi = partition(a, ll, rr - 1);
            st.emplace(mi + 1, rr);
            st.emplace(ll, mi);
        }
    }

    // a[lo, hi)
    static void merge(int * arr, int lo, int mi, int hi) 
    {   
        int * a = arr + lo;

        int * b = new int [mi - lo];
        int lb = mi - lo;
        for (int i = 0; i != mi - lo; ++i) b[i] = a[i];

        int * c = arr + mi;
        int lc = hi - mi;

        for (int i = 0, j = 0, k = 0; j < lb || k < lc; )
        {
            if (j < lb && (lc <= k || b[j] <= c[k])) a[i++] = b[j++];
            if (k < lc && (lb <= j || c[k] <  b[j])) a[i++] = c[k++];
        }

        delete [] b;
    }

    // a[lo, hi)
    static void mergeSort(int * arr, int lo, int hi)
    {
        if (hi < lo + 2) return;

        int mi = lo + ((hi - lo) >> 1);
        mergeSort(arr, lo, mi);
        mergeSort(arr, mi, hi);

        merge(arr, lo, mi, hi);
    }

    static void mergeSortIterative(int * arr, int lo, int hi)
    {
        if (hi < lo + 2) return;
        
        arr += lo;
        int n = hi - lo;

        // Invoke merge routine sequentially along arr
        // with granularity 1, 2, 4, 8, ...
        for (int step = 1, ll = 0, mi, rr; step < n; step <<= 1, ll = 0)
        {
            while (ll < n)
            {
                mi = ll + step;
                if (n - 1 < mi) break;  // Left part is sorted already. 
                rr = std::min(mi + step, n);
                merge(arr, ll, mi, rr);
                ll = rr;
            }
        }
    }
};
