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
    int maxDepth(TreeNode * root)
    {
        if (!root) return 0;
        if (!root->left && !root->right) return 1;

        std::stack<std::pair<TreeNode *, int>> st;
        TreeNode * head = root;
        st.emplace(head, 1);

        int maxDepth = -1;

        // Pre-Order Traversal. 
        while (!st.empty())
        {
            auto [cur, curDepth] = st.top();
            st.pop();
            maxDepth = std::max(maxDepth, curDepth);
            if (cur->right) st.emplace(cur->right, curDepth + 1);
            if (cur->left) st.emplace(cur->left, curDepth + 1);
        }

        // // Post-Order Traversal. 
        // while (!st.empty())
        // {
        //     auto [cur, curDepth] = st.top();
        // 
        //     if (cur->left && head != cur->left && head != cur->right)
        //     {
        //         st.emplace(cur->left, curDepth + 1);
        //     }
        //     else if (cur->right && head != cur->right)
        //     {
        //         st.emplace(cur->right, curDepth + 1);
        //     }
        //     else
        //     {
        //         maxDepth = std::max(maxDepth, curDepth);
        //         head = cur;
        //         st.pop();
        //     }
        // }

        return maxDepth;
    }
};