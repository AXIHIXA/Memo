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
    bool isSymmetric(TreeNode * root)
    {
        if (!root || (!root->left && !root->right)) return true;

        std::queue<TreeNode *> qq;
        qq.emplace(root);
        qq.emplace(root);

        while (!qq.empty())
        {
            TreeNode * t1 = qq.front();
            qq.pop();
            TreeNode * t2 = qq.front();
            qq.pop();

            if (!t1 && !t2) continue;
            if (!t1 || !t2) return false;
            if (t1->val != t2->val) return false;

            qq.emplace(t1->left);
            qq.emplace(t2->right);
            qq.emplace(t1->right);
            qq.emplace(t2->left);
        }

        return true;
    }
};