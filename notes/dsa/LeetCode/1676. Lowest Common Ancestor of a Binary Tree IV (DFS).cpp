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
        return dfs(root, nodes);
    }

private:
    TreeNode * dfs(TreeNode * root, const std::vector<TreeNode *> & nodes)
    {
        if (!root) return nullptr;
        if (std::find(nodes.cbegin(), nodes.cend(), root) != nodes.cend()) return root;

        TreeNode * left = dfs(root->left, nodes);
        TreeNode * right = dfs(root->right, nodes);
        if (left && right) return root;

        return left ? left : right;
    }
};