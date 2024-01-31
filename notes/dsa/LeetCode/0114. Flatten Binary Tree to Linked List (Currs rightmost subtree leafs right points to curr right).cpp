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
    void flatten(TreeNode * root)
    {
        if (!root) return;

        for (TreeNode * curr = root; curr; curr = curr->right)
        {
            if (curr->left)
            {
                TreeNode * rightmost = curr->left;
                while (rightmost->right) rightmost = rightmost->right;

                rightmost->right = curr->right;
                curr->right = curr->left;
                curr->left = nullptr;
            }
        }
    }

private:
    void flattenRecusrion(TreeNode * root) 
    {
        if (!root) return;
        if (root->right) flattenRecusrion(root->right);
        if (root->left) flattenRecusrion(root->left);
        root->right = temp;
        root->left = nullptr;
        temp = root;
    }

    TreeNode * temp = nullptr;
};