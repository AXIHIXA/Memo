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
    TreeNode * buildTree(std::vector<int> & inorder, std::vector<int> & postorder)
    {
        postorderIndex = postorder.size() - 1;
        if (postorderIndex == 0) return new TreeNode(postorder.back());
        for (int i = 0; i < inorder.size(); ++i) hm.emplace(inorder[i], i);
        return buildTree(postorder, 0, postorder.size());
    }

private:
    TreeNode * buildTree(const std::vector<int> & postorder, int left, int right)
    {
        if (right < left + 1) return nullptr;

        int v = postorder[postorderIndex--];
        TreeNode * root = new TreeNode(v);

        root->right = buildTree(postorder, hm.at(v) + 1, right);
        root->left = buildTree(postorder, left, hm.at(v));
        
        return root;
    }

    std::unordered_map<int, int> hm;
    int postorderIndex {-1};
};