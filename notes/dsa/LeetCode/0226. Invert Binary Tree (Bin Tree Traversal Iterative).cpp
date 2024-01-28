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
    TreeNode * invertTree(TreeNode * root)
    {
        if (!root || (!root->left && !root->right)) return root;
        
        std::stack<TreeNode *> st;
        st.emplace(root);

        while (!st.empty())
        {
            TreeNode * cur = st.top();
            st.pop();
            std::swap(cur->left, cur->right);
            if (cur->right) st.emplace(cur->right);
            if (cur->left) st.emplace(cur->left);
        }

        return root;
    }
};