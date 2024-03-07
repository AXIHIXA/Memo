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
    bool isBalanced(TreeNode * root)
    {
        bool ans = true;
        
        std::function<int (TreeNode *)> dfs = [&dfs, &ans](TreeNode * p) -> int
        {
            if (!p || !ans) return 0;
            int ll = dfs(p->left);
            int rr = dfs(p->right);
            if (1 < std::abs(ll - rr)) ans = false;
            return 1 + std::max(ll, rr);
        };

        dfs(root);

        return ans;
    }
};