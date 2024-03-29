class Solution
{
public:
    long long countSubarrays(std::vector<int> & nums, int k)
    {
        auto n = static_cast<const int>(nums.size());
        int maxi = *std::max_element(nums.cbegin(), nums.cend());

        long long ans = 0LL;

        for (int ll = 0, rr = 0, maxiFreq = 0; rr < n; ++rr)
        {
            maxiFreq += (nums[rr] == maxi);

            while (ll <= rr && k <= maxiFreq)
            {
                maxiFreq -= (nums[ll++] == maxi);
            }

            // A valid subarray could start from [0...ll) and end at rr. 
            ans += ll;
        }

        return ans;
    }

private:
    static long long atMostKMinusOne(std::vector<int> & nums, int k)
    {
        auto n = static_cast<const int>(nums.size());
        int maxi = *std::max_element(nums.cbegin(), nums.cend());

        long long numSubarrays = ((1LL + n) * n) >> 1LL;
        long long atMostKMinus1 = 0LL;

        for (int ll = 0, rr = 0, maxiFreq = 0; rr < n; ++rr)
        {
            maxiFreq += (nums[rr] == maxi);

            while (ll <= rr && k - 1 < maxiFreq)
            {
                maxiFreq -= (nums[ll++] == maxi);
            }

            atMostKMinus1 += rr - ll + 1;
        }

        return numSubarrays - atMostKMinus1;
    }
};