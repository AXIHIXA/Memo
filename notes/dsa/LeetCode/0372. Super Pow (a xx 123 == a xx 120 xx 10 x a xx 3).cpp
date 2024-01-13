class Solution 
{
public:
    int superPow(int a, vector<int> & b) 
    {
        if (b.empty()) return 1;

        int d = b.back();
        b.pop_back();

        return powm(superPow(a, b), 10) * powm(a, d) % kModulo;
    }

private:
    static constexpr int kModulo = 1337;

    int powm(int a, int b)
    {
        a %= kModulo;

        int r = 1;

        while (b--)
        {
            r = (r * a) % kModulo;
        }

        return r;
    }
};