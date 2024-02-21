class Bitset
{
public:
    Bitset(int size) : bitLength(size), data((size + kBitsPerInt - 1) / kBitsPerInt)
    {
        
    }
    
    void fix(int idx)
    {
        int i = idx / kBitsPerInt;
        int j = idx & (kBitsPerInt - 1);
        unsigned int mask = (1U << j);

        if (flipped)
        {
            if (data[i] & mask)
            {
                ++bitCount;
                data[i] ^= mask;
            }
        }
        else
        {
            if (!(data[i] & mask))
            {
                ++bitCount;
                data[i] ^= mask;
            }
        }
    }
    
    void unfix(int idx)
    {
        int i = idx / kBitsPerInt;
        int j = idx & (kBitsPerInt - 1);
        unsigned int mask = (1U << j);

        if (flipped)
        {
            if (!(data[i] & mask))
            {
                --bitCount;
                data[i] ^= mask;
            }
        }
        else
        {
            if (data[i] & mask)
            {
                --bitCount;
                data[i] ^= mask;
            }
        }
    }
    
    void flip()
    {
        flipped ^= 1;
        bitCount = bitLength - bitCount;
    }
    
    bool all()
    {
        return bitCount == bitLength;
    }
    
    bool one()
    {
        return bitCount;
    }
    
    int count()
    {
        return bitCount;
    }
    
    std::string toString()
    {
        std::string ans(bitLength, '\0');

        for (int bit = 0, i, j, v; bit < bitLength; ++bit)
        {
            i = bit / kBitsPerInt;
            j = bit & (kBitsPerInt - 1);
            v = (data[i] & (1 << j)) != 0;
            ans[bit] = '0' + (flipped ? !v : v);
        }

        return ans;
    }

private:
    static constexpr int kBitsPerInt = 32;

    const int bitLength;
    int bitCount = 0;
    bool flipped = false;
    std::vector<unsigned int> data;
};

/**
 * Your Bitset object will be instantiated and called as such:
 * Bitset* obj = new Bitset(size);
 * obj->fix(idx);
 * obj->unfix(idx);
 * obj->flip();
 * bool param_4 = obj->all();
 * bool param_5 = obj->one();
 * int param_6 = obj->count();
 * string param_7 = obj->toString();
 */