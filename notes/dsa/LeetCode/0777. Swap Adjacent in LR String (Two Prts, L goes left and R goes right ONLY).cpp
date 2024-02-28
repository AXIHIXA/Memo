class Solution
{
public:
    bool canTransform(std::string start, std::string end)
    {
        auto n = static_cast<const int>(start.size());

        int i = 0;
        int j = 0;

        // L could move towards left across X;
        // R could move towards right across X;
        // L/R could NOT move across each other.
        // Thus k-th L must in start must come after k-th L in end, 
        // and there MUST NOT be any R between start-L's pos and end-L's pos. 
        while (i < n || j < n)
        {
            while (i < n && start[i] == 'X') ++i;
            while (j < n && end[j] == 'X') ++j;
            if (i == n || j == n) return i == j;
            if (start[i] != end[j]) return false;
            if (start[i] == 'L' && i < j) return false;
            if (start[i] == 'R' && j < i) return false;
            ++i, ++j;
        }

        return i == j;
    }
};