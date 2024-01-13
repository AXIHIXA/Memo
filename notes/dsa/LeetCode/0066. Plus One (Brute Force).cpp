class Solution 
{
public:
    vector<int> plusOne(vector<int> & digits) 
    {
        int increment = 1;
        
        for (int i = digits.size() - 1; -1 < i; --i)
        {
            digits[i] += increment;

            if (9 < digits[i])
            {
                increment = 1;
                digits[i] -= 10;
            }
            else
            {
                increment = 0;
            }
        }

        if (increment)
        {
            digits.insert(digits.begin(), 1);
        }

        return digits;
    }
};