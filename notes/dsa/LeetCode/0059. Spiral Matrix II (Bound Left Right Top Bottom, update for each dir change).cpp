class Solution
{
public:
    std::vector<std::vector<int>> generateMatrix(int n)
    {
        std::vector ans(n, std::vector<int>(n, 0));
        
        int left = 0;
        int right = n - 1;
        int top = 0;
        int bottom = n - 1;
        int a = 1;

        while (left <= right || top <= bottom) 
        {
            for (int i = left; i <= right; i++) 
            {
                ans[top][i] = a++;
            }

            top++;

            for (int i = top; i <= bottom; i++) 
            {
                ans[i][right] = a++;
            }

            right--;

            for (int i = right; i >= left; i--) 
            {
                ans[bottom][i] = a++;
            }

            bottom--;

            for (int i = bottom; i >= top; i--) 
            {
                ans[i][left] = a++;
            }

            left++;
        }

        return ans;
    }
};