class Solution
{
public:
    int longestSubarray(std::vector<int> & nums, int limit)
    {
        auto n = static_cast<const int>(nums.size());

        int ans = 0;

        maxDeq[0] = 0;
        minDeq[0] = 0;
        int maxDl = 0, maxDr = 1;
        int minDl = 0, minDr = 1;

        for (int ll = 0, rr = 0; ll < n; ++ll)
        {
            // Extend window as long as it's valid.
            while (rr < n && nums[maxDeq[maxDl]] - nums[minDeq[minDl]] <= limit)
            {
                ans = std::max(ans, rr - ll + 1);

                if (rr == n - 1)
                {
                    break;
                }

                ++rr;

                while (maxDl < maxDr && nums[maxDeq[maxDr - 1]] <= nums[rr])
                {
                    --maxDr;
                }

                maxDeq[maxDr++] = rr;

                while (minDl < minDr && nums[rr] <= nums[minDeq[minDr - 1]])
                {
                    --minDr;
                }

                minDeq[minDr++] = rr;
            }

            // Slide window rightwards by one unit.
            while (maxDl < maxDr && maxDeq[maxDl] < ll + 1)
            {
                ++maxDl;
            }

            while (minDl < minDr && minDeq[minDl] < ll + 1)
            {
                ++minDl;
            }
        }

        return ans;
    }

private:
    static constexpr int kDeqSize = 100'010;
    static std::array<int, kDeqSize> maxDeq;
    static std::array<int, kDeqSize> minDeq;
};

std::array<int, Solution::kDeqSize> Solution::maxDeq = {0};
std::array<int, Solution::kDeqSize> Solution::minDeq = {0};