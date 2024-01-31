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
    bool hasPathSum(TreeNode * root, int targetSum)
    {
        if (!root) return false;
        
        std::stack<std::pair<TreeNode *, int>> st;
        st.emplace(root, 0);

        while (!st.empty())
        {
            auto [curr, sum] = st.top();
            st.pop();
            sum += curr->val;
            if (sum == targetSum && !curr->left && !curr->right) return true;

            if (curr->right) st.emplace(curr->right, sum);
            if (curr->left) st.emplace(curr->left, sum);
        }

        return false;
    }
};
