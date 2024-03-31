class Solution
{
public:
    int maxValue(int n, int index, int maxSum)
    {
        if (maxSum == n)
        {
            return 1;
        }
        
        int lo = 1;
        int hi = maxSum - n + 2;  // Can not take hi. 
        int ans;

        while (lo < hi)
        {
            int mi = lo + ((hi - lo) >> 1);

            int ll = std::max(0, index - mi + 1);
            long long nll = mi + ll - index;

            int rr = std::min(index + mi - 1, n - 1);
            long long nrr = mi + index - rr;

            long long sum = (ll - 0) + ((nll + mi) * (index - ll + 1) >> 1) + 
                      ((mi + nrr) * (rr - index + 1) >> 1) - mi + (n - 1 - rr);
            
            // std::printf(
            //         "mi = %d, ll = %d, nll = %d, rr = %d, nrr = %d, sum = %d\n", 
            //         mi, ll, nll, rr, nrr, sum);
            
            if (sum <= maxSum)
            {
                lo = mi + 1;
                ans = mi;
            }
            else
            {
                hi = mi;
            }
        }

        return ans;
    }
};

// ll .... index .... rr
// x   .... mi   .... y   
// index - ll == mi - x  --> x == mi + ll - index
// rr - index == mi - y

// 0 1 2 3
// 1 1 2 1
// 1 2 3 2