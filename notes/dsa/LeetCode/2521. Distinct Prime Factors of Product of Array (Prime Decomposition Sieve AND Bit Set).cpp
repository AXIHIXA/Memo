class Solution
{
public:
    int distinctPrimeFactors(std::vector<int> & nums)
    {
        static constexpr int kMaxNum = 1000;
        std::bitset<kMaxNum + 1> bs;

        for (int x : nums)
        {
            for (int f = 2; f * f <= x; )
            {
                if (x % f == 0)
                {
                    do x /= f; while (x % f == 0);
                    bs.set(f);
                }
                else
                {
                    ++f;
                }
            }

            bs.set(x);
        }

        bs.reset(1);

        return bs.count();
    }
};