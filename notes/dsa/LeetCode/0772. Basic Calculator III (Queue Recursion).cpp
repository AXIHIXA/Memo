class Solution
{
public:
    int calculate(std::string s)
    {
        std::queue<char> q;
        for (char c : s) q.push(c); 
        q.push('+');
        return calculate(q);
    }

private:
    static int calculate(std::queue<char> & q)
    {
        char preOp = '+';
        int num = 0, prev = 0, sum = 0;

        while (!q.empty())
        {
            char c = q.front();
            q.pop();

            if (std::isdigit(c)) 
            {
                num = num * 10 + (c - '0');
            } 
            else if (c == '(') 
            {
                num = calculate(q);
            } 
            else 
            {
                switch (preOp)
                {
                    case '+': 
                    {
                        sum += prev;
                        prev = num;
                        break;
                    }
                    case '-':
                    {
                        sum += prev;
                        prev = num * -1;
                        break;
                    }
                    case '*':
                    {
                        prev *= num;
                        break;
                    }
                    case '/':
                    {
                        prev /= num;
                        break;
                    }
                }
                
                if (c == ')')
                {
                    break;
                }

                num = 0;
                preOp = c;
            }
        }

        return sum + prev;
    }
};