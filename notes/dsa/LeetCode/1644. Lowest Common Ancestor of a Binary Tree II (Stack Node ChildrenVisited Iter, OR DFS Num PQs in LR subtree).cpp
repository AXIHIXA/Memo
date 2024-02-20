/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution
{
public:
    TreeNode * lowestCommonAncestor(TreeNode * root, TreeNode * p, TreeNode * q)
    {
        TreeNode * ans;
        if (dfs(root, p, q, &ans) < 2) return nullptr;
        return ans;
    }
    
    int dfs(TreeNode * node, TreeNode * p, TreeNode * q, TreeNode ** ans)
    {
        if (!node) return 0;

        int left = dfs(node->left, p, q, ans);
        int right = dfs(node->right, p, q, ans);

        if (left == 2 || right == 2) return 2;

        int flag = left + right;
        if (node == p || node == q) ++flag;

        if (flag == 2) *ans = node;

        return flag;
    }
    
private:
    // Iterative by stack<pair<Node, childrenVisited>>. 
    TreeNode * lowestCommonAncestorIterative(TreeNode * root, TreeNode * p, TreeNode * q)
    {
        if (!root) return nullptr;

        // Number of (p, q)s found; 
        // Current lca candidate.
        int found = 0;
        TreeNode * lca = nullptr;

        // {Node, Number of children visited}
        std::stack<std::pair<TreeNode *, int>> st;
        st.emplace(root, 0);

        while (!st.empty())
        {
            auto & [node, childrenVisited] = st.top();

            if (childrenVisited < 2)
            {
                if (childrenVisited == 0)
                {
                    if (node->left)
                    {
                        st.emplace(node->left, 0);
                    }

                    if (node == p || node == q)
                    {
                        if (++found == 2)
                        {
                            return lca;
                        }

                        lca = node;
                    }
                }
                else
                {
                    if (node->right)
                    {
                        st.emplace(node->right, 0);
                    }
                }

                ++childrenVisited;
            }
            else
            {
                st.pop();

                if (lca == node && !st.empty())
                {
                    lca = st.top().first;
                }
            }
        }

        return found == 2 ? lca : nullptr;
    }
};