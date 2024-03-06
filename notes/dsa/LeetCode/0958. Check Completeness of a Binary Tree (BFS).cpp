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
    bool isCompleteTree(TreeNode * root)
    {
        que[0] = root;
        bool bottom = false;

        for (int ll = 0, rr = 1; ll < rr; )
        {
            TreeNode * p = que[ll++];

            if ((!p->left && p->right) || (bottom && (p->left || p->right)))
            {
                return false;
            }

            bottom = !p->left || !p->right;

            if (p->left) que[rr++] = p->left;
            if (p->right) que[rr++] = p->right;
        }

        return true;
    }

private:
    static std::array<TreeNode *, 110> que;
};

std::array<TreeNode *, 110> Solution::que;
