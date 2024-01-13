class Solution 
{
public:
    double findMedianSortedArrays(vector<int> & nums1, vector<int> & nums2) 
    {
        return nums1.size() < nums2.size() ? f(nums1, nums2) : f(nums2, nums1);
    }

private:
    static constexpr int kIntMax = numeric_limits<int>::max();
    static constexpr int kIntMin = numeric_limits<int>::min();

    double f(vector<int> & a, vector<int> & b)
    {
        int m = a.size(), n = b.size();
        int ll = 0, rr = m;

        while (ll <= rr)
        {
            int mA = (ll + rr) >> 1;
            int mB = ((m + n + 1) >> 1) - mA;

            int maxLa = (mA == 0) ? kIntMin : a[mA - 1];
            int minRa = (mA == m) ? kIntMax : a[mA];
            int maxLb = (mB == 0) ? kIntMin : b[mB - 1];
            int minRb = (mB == n) ? kIntMax : b[mB];

            if (maxLa <= minRb and maxLb <= minRa)
            {
                if ((m + n) & 1)
                {
                    return max(maxLa, maxLb);
                }
                else
                {
                    return static_cast<double>(max(maxLa, maxLb) + min(minRa, minRb)) / 2.0;
                }
            }
            else if (minRb < maxLa)
            {
                rr = mA - 1;
            }
            else
            {
                ll = mA + 1;
            }
        }

        return 0.0;
    }
};