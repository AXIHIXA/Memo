class Solution
{
public:
    int minAbsDifference(std::vector<int> & nums, int goal)
    {
        int posSum = 0, negSum = 0;

        for (int x : nums)
        {
            if (0 <= x)
            {
                posSum += x;
            }
            else
            {
                negSum += x;
            }
        }

        if (posSum < goal)
        {
            return goal - posSum;
        }

        if (goal < negSum)
        {
            return negSum - goal;
        }
        
        arr = nums.data();
        n = static_cast<int>(nums.size());
        
        lsHi = 0;
        rsHi = 0;

        f(0, n >> 1, 0, ls.data(), &lsHi);
        f(n >> 1, n, 0, rs.data(), &rsHi);

        std::sort(ls.begin(), ls.begin() + lsHi);
        std::sort(rs.begin(), rs.begin() + rsHi);

        int ans = std::numeric_limits<int>::max();

        for (int i = 0, j = rsHi - 1; i < lsHi; ++i)
        {
            while (0 < j && 
                   std::abs(goal - ls[i] - rs[j - 1]) <= 
                   std::abs(goal - ls[i] - rs[j]))
            {
				--j;
			}

			ans = std::min(ans, std::abs(goal - ls[i] - rs[j]));
        }

        return ans;
    }

private:
    void f(int b, int e, int cur, int * res, int * hi)
    {
        if (b == e)
        {
            res[(*hi)++] = cur;
        }
        else
        {
            f(b + 1, e, cur, res, hi);
            f(b + 1, e, cur + arr[b], res, hi);
        }
    }

private:
    static constexpr int kMaxM = 1 << 20;

    static std::array<int, kMaxM> ls;
    static std::array<int, kMaxM> rs;

    static int lsHi;
    static int rsHi;

    int * arr = nullptr;
    int n = 0;
};

std::array<int, Solution::kMaxM> Solution::ls = {0};
std::array<int, Solution::kMaxM> Solution::rs = {0};

int Solution::lsHi = 0;
int Solution::rsHi = 0;
