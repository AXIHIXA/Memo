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
    // By DFS. 
    int findBottomLeftValue(TreeNode * root)
    {
        std::stack<std::pair<TreeNode *, int>> st;
        st.emplace(root, 1);

        int maxDepth = 0;
        int ans = -1;

        while (!st.empty())
        {
            auto [curr, currDepth] = st.top();
            st.pop();

            if (maxDepth < currDepth)
            {
                maxDepth = currDepth;
                ans = curr->val;
            }

            if (curr->right) st.emplace(curr->right, currDepth + 1);
            if (curr->left) st.emplace(curr->left, currDepth + 1);
        }

        return ans;
    }

    // By BFS. 
    int findBottomLeftValueBFS(TreeNode * root)
    {
        std::queue<std::pair<TreeNode *, int>> qu;
        qu.emplace(root, 1);

        int maxDepth = 0;
        int ans = -1;

        while (!qu.empty())
        {
            auto [curr, currDepth] = qu.front();
            qu.pop();

            if (maxDepth < currDepth)
            {
                maxDepth = currDepth;
                ans = curr->val;
            }

            if (curr->left) qu.emplace(curr->left, currDepth + 1);
            if (curr->right) qu.emplace(curr->right, currDepth + 1);
        }

        return ans;
    }
};