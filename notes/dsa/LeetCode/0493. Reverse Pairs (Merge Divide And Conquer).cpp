class Solution 
{
public:
    int reversePairs(vector<int> & nums) 
    {
        return count(nums.data(), 0, nums.size());
    }

private:
    static int count(int * arr, int lo, int hi) 
    {
        if (hi < lo + 2) return 0;
        int mi = (lo + hi) / 2;
        return count(arr, lo, mi) + count(arr, mi, hi) + mergeCount(arr, lo, mi, hi);
    }

    static int mergeCount(int * arr, int lo, int mi, int hi) 
    {
        // 统计部分
        int ans = 0;
        
        for (int i = lo, j = mi; i < mi; ++i) 
        {
            while (j < hi && 2 * static_cast<long long>(arr[j]) < static_cast<long long>(arr[i])) ++j;
            ans += j - mi;
        }

        // 正常merge
        merge(arr, lo, mi, hi);

        return ans;
    }

    static void merge(int * arr, int lo, int mi, int hi) 
    {
        int * a = arr + lo;

        int * b = buf;
        int lb = mi - lo;
        for (int i = 0; i != mi - lo; ++i) b[i] = a[i];

        int * c = arr + mi;
        int lc = hi - mi;

        for (int i = 0, j = 0, k = 0; j < lb || k < lc; )
        {
            if (j < lb && (lc <= k || b[j] <= c[k])) a[i++] = b[j++];
            if (k < lc && (lb <= j || c[k] <  b[j])) a[i++] = c[k++];
        }
    }

    static int buf[500001];
};

int Solution::buf[500001];
