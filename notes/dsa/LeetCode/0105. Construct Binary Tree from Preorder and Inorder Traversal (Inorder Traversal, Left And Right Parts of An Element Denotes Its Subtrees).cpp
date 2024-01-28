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
    TreeNode * buildTree(std::vector<int> & preorder, std::vector<int> & inorder)
    {
        if (preorder.size() == 1) return new TreeNode(preorder.front());
        for (int i = 0; i < inorder.size(); ++i) hm.emplace(inorder[i], i);
        return buildTree(preorder, 0, preorder.size());
    }

private:
    TreeNode * buildTree(const std::vector<int> & preorder, int left, int right)
    {
        if (right < left + 1) return nullptr;

        int v = preorder[preorderIndex++];
        TreeNode * root = new TreeNode(v);

        root->left = buildTree(preorder, left, hm.at(v));
        root->right = buildTree(preorder, hm.at(v) + 1, right);

        return root;
    }

    std::unordered_map<int, int> hm;
    int preorderIndex {0};
};