class Solution 
{
public:
    string multiply(string num1, string num2) 
    {
        if (num1 == "0" or num2 == "0")
        {
            return "0";
        }
        
        string res(num1.size() + num2.size(), '0');

        for (int i = num1.size() - 1; -1 < i; --i)
        {
            for (int j = num2.size() - 1; -1 < j; --j)
            {
                int sum = (num1[i] - '0') * (num2[j] - '0') + (res[i + j + 1] - '0');
                res[i + j] += sum / 10;
                res[i + j + 1] = '0' + sum % 10;
            }
        }

        return res.substr(res.find_first_not_of('0'));
    }
};