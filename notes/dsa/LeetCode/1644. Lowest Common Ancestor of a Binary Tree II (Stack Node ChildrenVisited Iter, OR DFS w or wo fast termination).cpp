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
        // (1) Recursive LCS DFS. 
    #if false
        TreeNode * ans = nullptr;
        int ret = dfs2(root, p, q, &ans);
        return ret == 2 ? ans : nullptr;
    #endif  // #if false
        
        // (2) First search for all then call fast-stop LCS DFS. 
    #if false
        int found = 0;
        std::stack<TreeNode *> st;
        st.emplace(root);

        while (!st.empty())
        {
            TreeNode * node = st.top();
            st.pop();
            if (node == p || node == q) ++found;
            if (node->right) st.emplace(node->right);
            if (node->left) st.emplace(node->left);
        }

        return found == 2 ? dfs1(root, p, q) : nullptr;
    #endif  // #if false

        // (3) Iterative version. 
    #if true
        return lowestCommonAncestorIterative(root, p, q);
    #endif  // #if true
    }
    
private:
    // Recursive LCS DFS with fast termination. 
    static TreeNode * dfs1(TreeNode * root, TreeNode * p, TreeNode * q)
    {
        if (!root) return nullptr;
        if (root == p || root == q) return root;

        TreeNode * left = dfs1(root->left, p, q);
        TreeNode * right = dfs1(root->right, p, q);
        if (left && right) return root;

        return left ? left : right;
    }

    // Recursive LCS DFS without fast termination, 
    // used when given children not guaranteed to exist. 
    static int dfs2(TreeNode * root, TreeNode * p, TreeNode * q, TreeNode ** ans)
    {
        if (!root) return 0;

        int left = dfs2(root->left, p, q, ans);
        int right = dfs2(root->right, p, q, ans);
        if (left == 2 || right == 2) return 2;

        int ret = left + right + (root == p || root == q);
        if (ret == 2) *ans = root;
        return ret;
    }

    // Iterative by stack<pair<Node, childrenVisited>>. 
    static TreeNode * lowestCommonAncestorIterative(TreeNode * root, TreeNode * p, TreeNode * q)
    {
        if (!root) return nullptr;

        TreeNode * ans = nullptr;
        int found = 0;

        std::stack<std::pair<TreeNode *, int>> st;
        st.emplace(root, 0);

        while (!st.empty())
        {
            auto & [node, childrenVisited] = st.top();
            // Do NOT pop here! Needed in stack until all children visited!

            if (childrenVisited < 2)
            {
                if (childrenVisited == 0)
                {
                    if (node == p || node == q)
                    {
                        if (++found == 2) return ans;
                        ans = node;
                    }

                    if (node->left) st.emplace(node->left, 0);
                }
                else  // childrenVisited == 1
                {
                    if (node->right) st.emplace(node->right, 0);
                }

                ++childrenVisited;
            }
            else  // childrenVisited == 2
            {
                st.pop();
                if (node == ans && !st.empty()) ans = st.top().first;
            }
        }

        return nullptr;
    }
};