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
    int sumNumbers(TreeNode * root)
    {
        if (!root) return 0;

        int ans = 0;
        int currNumber = 0;
        TreeNode * prev = nullptr;

        // Morris Preorder Traversal -- O(1) Space.
        // Prev is currrent node's left's (root->left's) rightmost subtree leaf, 
        // which is end of the left subtree traversal. 
        // If there is a left child, then compute prev. 
        // If there is no link prev->right == root --> set it.
        // If there is a link prev->right == root --> break it.
        while (root)
        {
            if (root->left)
            {
                prev = root->left;
                int steps = 1;

                while (prev->right && prev->right != root)
                {
                    prev = prev->right;
                    ++steps;
                }

                if (!prev->right)
                {
                    currNumber = currNumber * 10 + root->val;
                    prev->right = root;
                    root = root->left;
                }
                else
                {
                    if (!prev->left) ans += currNumber;
                    for (int i = 0; i < steps; ++i) currNumber /= 10;
                    prev->right = nullptr;
                    root = root->right;
                }
            }
            else
            {
                // If there is no left child, then just go right.
                currNumber = currNumber * 10 + root->val;
                if (!root->right) ans += currNumber;
                root = root->right;
            }
        }

        return ans;
    }
};