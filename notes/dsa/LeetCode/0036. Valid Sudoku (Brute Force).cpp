class Solution 
{
public:
    bool isValidSudoku(vector<vector<char>> & board) 
    {
        array<bool, kSz> arr {false};

        for (const auto & row : board)
        {
            for (char c : row)
            {
                if (c == '.')
                {
                    continue;
                }

                // cout << c << ' ' << c - '1' << '\n';

                if (arr[c - '1'])
                {
                    return false;
                }
                else
                {
                    arr[c - '1'] = true;
                }
            }

            memset(arr.data(), 0, sizeof(bool) * kSz);
        }

        for (int j = 0; j != kSz; ++j)
        {
            for (int i = 0; i != kSz; ++i)
            {
                if (board[i][j] == '.')
                {
                    continue;
                }
                
                if (arr[board[i][j] - '1'])
                {
                    return false;
                }
                else
                {
                    arr[board[i][j] - '1'] = true;
                }
            }

            memset(arr.data(), 0, sizeof(bool) * kSz);
        }

        for (int i0 = 0; i0 != kSz; i0 += 3)
        {
            for (int j0 = 0; j0 != kSz; j0 += 3)
            {
                for (int i = i0; i != i0 + 3; ++i)
                {
                    for (int j = j0; j != j0 + 3; ++j)
                    {
                        if (board[i][j] == '.')
                        {
                            continue;
                        }
                        
                        if (arr[board[i][j] - '1'])
                        {
                            return false;
                        }
                        else
                        {
                            arr[board[i][j] - '1'] = true;
                        }
                    }
                }

                memset(arr.data(), 0, sizeof(bool) * kSz);
            }
        }

        return true;
    }

private:
    static constexpr int kSz = 9;
};