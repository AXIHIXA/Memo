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
    bool isValidBST(TreeNode * root)
    {
        bool ans = true;
        
        std::function<void (TreeNode *, long long, long long)> dfs = 
        [&dfs, &ans](TreeNode * p, long long lo, long long hi)
        {
            if (!ans)
            {
                return;
            }

            if (p->val <= lo || hi <= p->val)
            {
                ans = false;
                return;
            }

            if (p->left) dfs(p->left, lo, p->val);
            if (p->right) dfs(p->right, p->val, hi);
        };

        dfs(root, std::numeric_limits<long long>::min(), std::numeric_limits<long long>::max());

        return ans;
    }
};