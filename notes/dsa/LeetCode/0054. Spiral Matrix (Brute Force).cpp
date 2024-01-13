class Solution 
{
public:
    enum Direction : int
    {
        kRight = 0, 
        kDown = 1, 
        kLeft = 2,
        kUp = 3, 
        kMaxDirection = 4,
    };

    vector<int> spiralOrder(vector<vector<int>> & matrix) 
    {
        int m = matrix.size(), n = matrix[0].size();

        vector<int> mRes(m * n);
        vector<vector<unsigned char>> visited(m, vector<unsigned char>(n, false));
        
        Direction dir = kRight;

        // i, j: matrix row, col indices
        // k: total elements visited
        for (int i = 0, j = 0, k = 0; k != m * n; ++k)
        {
            mRes[k] = matrix[i][j];
            visited[i][j] = true;
            
            switch (dir)
            {
                case kRight:
                {
                    if (j < n - 1 and !visited[i][j + 1])
                    {
                        ++j;
                    }
                    else
                    {
                        dir = kDown;
                        ++i;
                    }

                    break;
                }
                case kDown:
                {
                    if (i < m - 1 and !visited[i + 1][j])
                    {
                        ++i;
                    }
                    else
                    {
                        dir = kLeft;
                        --j;
                    }
                    
                    break;
                }
                case kLeft:
                {
                    if (0 < j and !visited[i][j - 1])
                    {
                        --j;
                    }
                    else
                    {
                        dir = kUp;
                        --i;
                    }
                    
                    break;
                }
                case kUp:
                {
                    if (0 < i and !visited[i - 1][j])
                    {
                        --i;
                    }
                    else
                    {
                        dir = kRight;
                        ++j;
                    }
                    
                    break;
                }
                default:
                {
                    throw invalid_argument("invalid direction enum value " + to_string(dir));
                }
            }
        }

        return mRes;
    }
};