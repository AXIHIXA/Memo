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
    std::vector<std::vector<int>> pathSum(TreeNode * root, int targetSum)
    {
        if (!root) return {};
        
        std::vector<std::vector<int>> ans;
        
        int sum = 0;
        std::vector<int> tmp;

        std::function<void (TreeNode *)> dfs = [&dfs, &ans, &sum, &tmp, targetSum](TreeNode * p)
        {
            tmp.emplace_back(p->val);
            sum += p->val;

            if (sum == targetSum && !p->left && !p->right) ans.push_back(tmp);

            if (p->left) dfs(p->left);
            if (p->right) dfs(p->right);

            sum -= p->val;
            tmp.pop_back();
        };

        dfs(root);

        return ans;
    }
};