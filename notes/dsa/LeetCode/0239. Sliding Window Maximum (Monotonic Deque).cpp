class Solution
{
public:
    std::vector<int> maxSlidingWindow(std::vector<int> & nums, int k)
    {
        auto n = static_cast<const int>(nums.size());
        std::vector<int> ans;
        ans.reserve(n - k + 1);

        // Array simulation of deque. 
        int dl = 0, dr = 0;

        for (int ll = 0, rr = 0; rr < n; ++rr)
        {
            while (dl < dr && nums[deq[dr - 1]] <= nums[rr])
            {
                --dr;
            }

            deq[dr++] = rr;

            while (ll <= rr && k < rr - ll + 1)
            {
                ++ll;

                while (dl < dr && deq[dl] < ll)
                {
                    ++dl;
                }
            }

            if (rr - ll + 1 == k)
            {
                ans.emplace_back(nums[deq[dl]]);
            }
        }

        return ans;
    }

private:
    static constexpr int kDeqSize = 100'010;
    static std::array<int, kDeqSize> deq;
};

std::array<int, Solution::kDeqSize> Solution::deq = {0};
