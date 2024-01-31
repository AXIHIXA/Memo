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
    int getMinimumDifference(TreeNode * root)
    {
        // The number of nodes in the tree is in the range [2, 1e4].
        int last = -1;
        int ans = std::numeric_limits<int>::max();
        
        while (root)
        {
            if (!root->left)
            {
                if (last != -1) ans = std::min(ans, std::abs(last - root->val));
                last = root->val;

                root = root->right;
            }
            else
            {
                TreeNode * prev = root->left;
                while (prev->right && prev->right != root) prev = prev->right;

                if (!prev->right)
                {
                    prev->right = root;
                    root = root->left;
                }
                else
                {
                    if (last != -1) ans = std::min(ans, std::abs(last - root->val));
                    last = root->val;
                    
                    prev->right = nullptr;
                    root = root->right;
                }
            }
        }

        return ans;
    }
};