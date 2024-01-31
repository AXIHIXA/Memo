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
    std::vector<std::vector<int>> zigzagLevelOrder(TreeNode * root)
    {
        if (!root) return {};
        std::vector<std::vector<int>> ans;

        std::queue<TreeNode *> qu;
        qu.emplace(root);

        while (!qu.empty())
        {
            int levelSize = qu.size();
            TreeNode * curr = nullptr;
            std::vector<int> level(levelSize);

            for (int i = 0; i < levelSize; ++i)
            {
                curr = qu.front();
                qu.pop();
                level[(ans.size() & 1) ? levelSize - i - 1 : i] = curr->val;
                if (curr->left) qu.emplace(curr->left);
                if (curr->right) qu.emplace(curr->right);
            }

            ans.push_back(std::move(level));
        }

        return ans;
    }
};