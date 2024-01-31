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
    std::vector<double> averageOfLevels(TreeNode * root)
    {
        if (!root) return {0.0};
        std::vector<double> ans;

        std::queue<TreeNode *> qu;
        qu.emplace(root);

        while (!qu.empty())
        {
            int levelSize = qu.size();
            ans.emplace_back(0);
            
            for (int i = 0; i < levelSize; ++i)
            {
                TreeNode * curr = qu.front();
                qu.pop();
                ans.back() += curr->val;
                if (curr->left) qu.emplace(curr->left);
                if (curr->right) qu.emplace(curr->right);
            }

            ans.back() /= static_cast<double>(levelSize);
        }

        return ans;
    }
};