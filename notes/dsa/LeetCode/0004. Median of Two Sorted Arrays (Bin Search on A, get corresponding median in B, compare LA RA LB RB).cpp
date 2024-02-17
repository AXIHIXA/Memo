class Solution 
{
public:
    double findMedianSortedArrays(std::vector<int> & nums1, std::vector<int> & nums2) 
    {
        return nums1.size() < nums2.size() ? f(nums1, nums2) : f(nums2, nums1);
    }

private:
    static constexpr int k_int_max = numeric_limits<int>::max();
    static constexpr int k_int_min = numeric_limits<int>::min();

    double f(std::vector<int> & a, std::vector<int> & b)
    {
        auto m = static_cast<const int>(a.size());
        auto n = static_cast<const int>(b.size());

        double ans = 0.0;

        for (int ll = 0, rr = m, ma, mb; ll <= rr; )
        {
            ma = ll + ((rr - ll) >> 1);
            mb = ((m + n + 1) >> 1) - ma;

            int max_la = 0 < ma ? a[ma - 1] : k_int_min;
            int min_ra = ma < m ? a[ma] : k_int_max;
            int max_lb = 0 < mb ? b[mb - 1] : k_int_min;
            int min_rb = mb < n ? b[mb] : k_int_max;

            if (max_la <= min_rb && max_lb <= min_ra)
            {
                ans = (m + n) & 1 ? 
                      std::max(max_la, max_lb) : 
                      0.5 * static_cast<double>(std::max(max_la, max_lb) + 
                                                std::min(min_ra, min_rb));
                break;
            }
            else if (min_rb < max_la)
            {
                rr = ma - 1;
            }
            else
            {
                ll = ma + 1;
            }
        }

        return ans;
    }
};