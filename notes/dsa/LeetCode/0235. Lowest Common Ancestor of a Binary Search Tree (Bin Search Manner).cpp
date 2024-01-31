/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */

class Solution
{
public:
    TreeNode * lowestCommonAncestor(TreeNode * root, TreeNode * p, TreeNode * q)
    {
        TreeNode * node = root;
        int pv = p->val;
        int qv = q->val;

        while (node)
        {
            int v = node->val;
            if (v < pv && v < qv)      node = node->right;
            else if (pv < v && qv < v) node = node->left;
            else                       return node;
        }

        return nullptr;
    }
};