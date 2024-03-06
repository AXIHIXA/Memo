class Solution
{
public:
    Solution()
    {
        static int init = collect();
    }

    int magicalString(int n)
    {
        return table[n];
    }

private:
    static int collect()
    {
        std::string s(100'010, '\0');
        s[0] = '0';
        s[1] = '1';
        table[1] = 1;

        for (int i = 1, j = 2; j < s.size(); )
        {
            if (s[i] == '1')
            {
                s[j] = s[j - 1] == '1' ? '2' : '1';
                table[j] = table[j - 1] + (s[j] == '1');
                ++i;
                ++j;
            }
            else  // s[i] == '2'
            {
                if (s[j - 1] == '1')
                {
                    s[j] = '1';
                    s[j + 1] = '2';
                    table[j] = table[j - 1] + 1;
                    table[j + 1] = table[j];
                }
                else
                {
                    s[j] = '2';
                    s[j + 1] = '1';
                    table[j] = table[j - 1];
                    table[j + 1] = table[j] + 1;
                }

                ++i;
                j += 2;
            }
        }
        
        return 0;
    }

private:
    static std::array<int, 100'020> table;
};

std::array<int, 100'020> Solution::table = {0};
