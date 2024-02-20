class Solution 
{
public:
    double findMedianSortedArrays(std::vector<int> & nums1, std::vector<int> & nums2) 
    {
        return nums1.size() < nums2.size() ? f(nums1, nums2) : f(nums2, nums1);
    }

private:
    static constexpr int kLargeInt = numeric_limits<int>::max();
    static constexpr int kSmallInt = numeric_limits<int>::min();
    
    // So we divide both a and b into two parts: 
    // [ LA <= a[mA - 1] ] <= [ a[mA] <= RA ]
    // [ LB <= b[mB - 1] ] <= [ a[mB] <= RB ]
    // We do binary search on a, and make sure 
    // #elements in LA and LB equals half size of a + b. 
    // Then as long as LA, LB lies on the left of RA and RB, 
    // the median will be a[mA], b[mB] and their predcessors. 
    // We also need to prepend dummies to a[-1] and a[a.size()] to make sure
    // the left/right parts could cover the whole a array; so does array b. 
    double f(std::vector<int> & a, std::vector<int> & b)
    {
        auto m = static_cast<const int>(a.size());
        auto n = static_cast<const int>(b.size());

        double ans = 0.0;

        for (int ll = 0, rr = m, mA, mB; ll <= rr; )
        {
            mA = ll + ((rr - ll) >> 1);
            mB = ((m + n + 1) >> 1) - mA;

            int maxLa = 0 < mA ? a[mA - 1] : kSmallInt;
            int minRa = mA < m ? a[mA] : kLargeInt;
            int maxLb = 0 < mB ? b[mB - 1] : kSmallInt;
            int minRb = mB < n ? b[mB] : kLargeInt;

            if (maxLa <= minRb && maxLb <= minRa)
            {
                ans = (m + n) & 1 ? 
                      std::max(maxLa, maxLb) : 
                      0.5 * static_cast<double>(std::max(maxLa, maxLb) + std::min(minRa, minRb));
                break;
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

        return ans;
    }
};