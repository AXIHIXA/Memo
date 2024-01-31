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
    TreeNode * bstFromPreorder(std::vector<int> & preorder)
    {
        int n = preorder.size();
        TreeNode * root = new TreeNode(preorder.front());
        if (n == 1) return root;

        std::stack<TreeNode *> st;
        st.emplace(root);

        for (int i = 1; i < n; ++i)
        {
            TreeNode * curr = st.top();
            TreeNode * child = new TreeNode(preorder[i]);

            while (!st.empty() && st.top()->val < child->val)
            {
                curr = st.top();
                st.pop();
            }

            if (curr->val < child->val) curr->right = child;
            else                        curr->left = child;
            st.emplace(child);
        }

        return root;

        // // Recursion Version. 
        // // Preorder traversal [Root LeftSubTree RightSubTree]. 
        // // RightSubTree's front element > Root, to be found be upper_bound. 
        // return build(preorder.data(), preorder.size());
    }

private:
    static TreeNode * build(int * a, int hi)
    {
        if (hi < 1) return nullptr;
        TreeNode * root = new TreeNode(a[0]);
        if (hi == 1) return root;

        int mi = std::upper_bound(a, a + hi, root->val) - a;

        root->left = build(a + 1, mi - 1);
        root->right = build(a + mi, hi - mi);

        return root;
    }
};