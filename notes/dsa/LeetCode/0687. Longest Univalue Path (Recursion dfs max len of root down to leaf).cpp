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
    int longestUnivaluePath(TreeNode * root)
    {
        if (!root)
        {
            return 0;
        }

        int ans = 0;

        dfs(root, &ans);

        return ans;
    }

private:
    static int dfs(TreeNode * root, int * ans)
    {
        if (!root)
        {
            return 0;
        }

        int l = dfs(root->left, ans);
        int r = dfs(root->right, ans);

        int ret = 0;
        int pathLen = 0;

        if (root->left && root->left->val == root->val)
        {
            ret = l + 1;
            pathLen += l + 1;
        }

        if (root->right && root->right->val == root->val)
        {
            ret = std::max(ret, r + 1);
            pathLen += r + 1;
        }

        *ans = std::max(*ans, pathLen);
        return ret;
    }
};