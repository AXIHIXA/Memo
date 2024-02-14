class Solution
{
public:
    int compress(std::vector<char> & chars)
    {
        auto n = static_cast<int>(chars.size());
        if (n == 1) return 1;

        int ll = 0, i = 0;

        while (i < n)
        {
            int groupLength = 1;
            while (i + groupLength < n && chars[i + groupLength] == chars[i]) ++groupLength;
            chars[ll++] = chars[i];

            if (1 < groupLength)
            {
                std::string num = std::to_string(groupLength);
                for (char c : num) chars[ll++] = c;
            }

            i += groupLength;
        }

        return ll;
    }
};