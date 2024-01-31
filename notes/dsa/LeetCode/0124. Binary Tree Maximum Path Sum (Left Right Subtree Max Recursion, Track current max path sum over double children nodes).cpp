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
    int maxPathSum(TreeNode * root)
    {
        if (!root) return 0;
        int ans = std::numeric_limits<int>::min();
        gainFromSubtree(root, &ans);
        return ans;
    }

private:
    // Only ONE node in the path may have both children selected. 
    // gainFromSubtree calculates subtree max paths with no double-children nodes. 
    int gainFromSubtree(TreeNode * root, int * ans)
    {
        if (!root) return 0;

        int left = std::max(gainFromSubtree(root->left, ans), 0);
        int right = std::max(gainFromSubtree(root->right, ans), 0);

        *ans = std::max({*ans, left + right + root->val});

        return std::max(left, right) + root->val;
    }
};