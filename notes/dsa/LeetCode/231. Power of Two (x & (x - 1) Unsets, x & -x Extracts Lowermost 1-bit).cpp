class Solution 
{
public:
    bool isPowerOfTwo(int n) 
    {
        if (n == 0)
        {
            return false;
        }

        // x & (-x) keeps only the rightmost 1-bit and unsets all other bits!
        //        x = [     ?     ] 1 [ 0 0 ... 0 ]     = [     ?     ] 1 [ 0 0 ... 0 ]
        //       -x = [    ~?     ] 0 [ 1 1 ... 1 ] + 1 = [    ~?     ] 1 [ 0 0 ... 0 ]
        // x & (-x) =                                   = [ 0 0 ... 0 ] 1 [ 0 0 ... 0 ]

        // x & (x - 1) unsets the rightmmost 1-bit!
        //           x = [ ? ] 1 [ 0 0 ... 0 ]
        //       x - 1 = [ ? ] 0 [ 1 1 ... 1 ]
        // x & (x - 1) = [ ? ] 0 [ 0 0 ... 0 ]

        long long x = n;

        // return (x & (-x)) == x;
        return not (x & (x - 1));
    }
};