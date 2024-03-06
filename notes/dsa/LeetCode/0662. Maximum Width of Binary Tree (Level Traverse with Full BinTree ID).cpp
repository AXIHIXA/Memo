/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution
{
public:
    int widthOfBinaryTree(TreeNode * root)
    {
        que[0] = {root, 0};

        int ans = 1;

        for (int ll = 0, rr = 1; ll < rr; )
        {
            int levelEnd = rr;
            ans = std::max(ans, que[rr - 1].second - que[ll].second + 1);
            
            for ( ; ll < levelEnd; ++ll)
            {
                auto [p, k] = que[ll];
                if (p->left) que[rr++] = {p->left, 1 + (k << 1)};
                if (p->right) que[rr++] = {p->right, 2 + (k << 1)};
            }
        }

        return ans;
    }

private:
    static std::array<std::pair<TreeNode *, int>, 3010> que;
};

std::array<std::pair<TreeNode *, int>, 3010> Solution::que;
