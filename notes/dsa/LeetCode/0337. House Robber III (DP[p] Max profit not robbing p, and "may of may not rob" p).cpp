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
    int rob(TreeNode * root)
    {
        // Max profit not robbing p, and "may of may not rob" p. 
        std::function<std::pair<int, int> (TreeNode *)> dfs = 
        [&dfs](TreeNode * p) -> std::pair<int, int>
        {
            if (!p) return {0, 0};

            auto ll = dfs(p->left);
            auto rr = dfs(p->right);

            int p0 = ll.second + rr.second;
            int p1 = p->val + ll.first + rr.first;

            return {p0, std::max(p0, p1)};
        };

        return dfs(root).second;
    }
};