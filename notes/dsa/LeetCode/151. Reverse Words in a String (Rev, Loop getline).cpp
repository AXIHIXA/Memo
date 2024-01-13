class Solution 
{
public:
    string reverseWords(string s) 
    {
        reverse(s.begin(), s.end());

        string res;
        res.reserve(s.size());
        istringstream sin(s);

        for (string line; getline(sin, line, ' '); )
        {
            if (line.empty()) continue;
            
            // cout << "[ " << line << " ]" << '\n';
            reverse(line.begin(), line.end());
            res += line + ' ';
        }

        return res.substr(0, res.size() - 1);
    }
};