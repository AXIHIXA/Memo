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
        auto n = static_cast<const int>(preorder.size());
        std::unordered_map<int, int> indexInInorder;
        for (int i = 0; i < n; ++i) indexInInorder.emplace(inorder[i], i);

        std::function<TreeNode * (int, int, int, int)> build = 
        [&preorder, &inorder, &indexInInorder, &build]
        (int l1, int r1, int l2, int r2) -> TreeNode *
        {
            if (r1 < l1 || r2 < l2) return nullptr;
            TreeNode * root = new TreeNode(preorder[l1]);
            int m2 = indexInInorder.at(preorder[l1]);
            int leftSubtreeSize = m2 - l2;
            root->left = build(l1 + 1, l1 + leftSubtreeSize, l2, m2 - 1);
            int rightSubtreeSize = r2 - m2;
            root->right = build(l1 + leftSubtreeSize + 1, r1, m2 + 1, r2);
            return root;
        };

        return build(0, n - 1, 0, n - 1);
    }
};