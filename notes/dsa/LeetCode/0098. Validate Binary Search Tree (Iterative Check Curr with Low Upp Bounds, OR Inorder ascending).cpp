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
    bool isValidBST(TreeNode * root)
    {
        if (!root) return true;

        // Node, lower limit, upper limit. 
        std::stack<std::tuple<TreeNode *, TreeNode *, TreeNode *>> st;
        st.emplace(root, nullptr, nullptr);

        while (!st.empty())
        {
            auto [curr, lo, hi] = st.top();
            st.pop();

            if (!curr) continue;
            if (lo && curr->val <= lo->val) return false;
            if (hi && hi->val <= curr->val) return false;

            st.emplace(curr->right, curr, hi);
            st.emplace(curr->left, lo, curr);
        }

        return true;
    }
};