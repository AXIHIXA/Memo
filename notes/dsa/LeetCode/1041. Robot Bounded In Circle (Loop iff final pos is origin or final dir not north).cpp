class Solution
{
public:
    bool isRobotBounded(std::string instructions)
    {
        // Will loop as long as final pos is origin or final direction is not north. 
        
        int x = 0;
        int y = 0;
        int d = 1;  // east, north, west, south. 

        for (char c : instructions)
        {
            switch (c)
            {
                case 'L':
                {
                    d = (d + 1) % 4;
                    break;
                }
                case 'R':
                {
                    d = (d - 1 + 4) % 4;
                    break;
                }
                case 'G':
                default:
                {
                    x += dx[d];
                    y += dy[d];
                    break;
                }
            }
        }

        return (x == 0 && y == 0) || d != 1;
    }

private:
    static constexpr std::array<int, 4> dx {1, 0, -1, 0};
    static constexpr std::array<int, 4> dy {0, 1, 0, -1};
};