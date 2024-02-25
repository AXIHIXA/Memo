class Solution
{
public:
    int checkRecord(int n)
    {
        if (n == 1) return 3;
        if (n == 2) return 8;

        // Rolling array. 
        // a0: #valid records ending with 'A' TODAY. 
        // a1: Yesterday's a0;
        // a2: a0 two days ago; 
        // a3: a0 three days ago. 
        long long a3 = 1LL;
        long long a2 = 1LL;
        long long a1 = 2LL;
        long long a0 = 0LL;

        long long p2 = 1LL;
        long long p1 = 3LL;
        long long p0 = 0LL;

        long long l1 = 3LL;
        long long l0 = 0LL;

        for (int i = 3; i <= n; ++i)
        {
            // Today's ".....P"s:
            // (1) Yesterday's  "....A" -> Today's "....A" + "P";
            // (2) Yesterday's  "....P" -> Today's "....P" + "P";
            // (3) Yesterday's  "....L" -> Today's "....L" + "P". 
            p0 = (a1 + p1 + l1) % p;

            // Today's ".....A"s:
            // (1) Yesterday's  "....A" -> Today's "....P" + "A";
            // (2) 2-days-ago's "...A"  -> Today's "...PL" + "A";
            // (3) 3-days-ago's "..A"   -> Today's "..PLL" + "A". 
            // Note that "...A" could only be obtained form non-"A" records 
            // so we have to record "...A"s for 3 days to get all patterns 
            // before this trailing "A" today. 
            a0 = (a1 + a2 + a3) % p;

            // Today's ".....L"s:
            // (1) Yesterday's  "....P" -> Today's "....P" + "L";
            // (2) Yesterday's  "....A" -> Today's "....A" + "L";
            // (3) 2-days-ago's "...P"  -> Today's "...PL" + "L";
            // (4) 2-days-ago's "...A"  -> Today's "...AL" + "L".
            l0 = (p1 + a1 + p2 + a2) % p;

            p2 = p1;
            p1 = p0;

            a3 = a2;
            a2 = a1;
            a1 = a0;

            l1 = l0;
        }

        return (a0 + p0 + l0) % p;
    }

private:
    static constexpr long long p = 1'000'000'007LL;
};