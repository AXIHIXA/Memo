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
    TreeNode * lowestCommonAncestor(TreeNode * root, std::vector<TreeNode *> & nodes)
    {
        if (!root || std::find(nodes.cbegin(), nodes.cend(), root) != nodes.cend())
        {
            return root;
        }

        TreeNode * ll = lowestCommonAncestor(root->left, nodes);
        TreeNode * rr = lowestCommonAncestor(root->right, nodes);

        if (ll && rr)
        {
            return root;
        }

        return ll ? ll : rr;
    }
};
