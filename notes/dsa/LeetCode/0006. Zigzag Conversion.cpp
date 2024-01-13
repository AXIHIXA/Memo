class Solution 
{
public:
    string convert(string s, int numRows) 
    {
        if (numRows == 1 or s.length() <= numRows)
        {
            return s;
        }

        string ans;

        for (int row = 0, n = s.size(), sn = 2 * numRows - 2; row != numRows; ++row)
        {
            int i = row;

            while (i < n)
            {
                ans += s[i];

                if (row != 0 and row != numRows - 1)
                {
                    int j = i + sn - 2 * row;

                    if (j < n)
                    {
                        ans += s[j];
                    }
                }

                i += sn;
            }
        }

        return ans;
    }
};