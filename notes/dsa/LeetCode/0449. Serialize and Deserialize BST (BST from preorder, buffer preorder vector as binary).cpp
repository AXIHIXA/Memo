/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Codec
{
public:

    // Encodes a tree to a single string.
    std::string serialize(TreeNode * root)
    {
        if (!root) return "";
        
        std::stack<TreeNode *> st;
        st.emplace(root);

        std::vector<int> preorder;

        while (!st.empty())
        {
            TreeNode * curr = st.top();
            st.pop();

            preorder.emplace_back(curr->val);
            if (curr->right) st.emplace(curr->right);
            if (curr->left) st.emplace(curr->left);
        }

        auto p = reinterpret_cast<char *>(preorder.data());
        return std::string(p, preorder.size() * sizeof(int));
    }

    // Decodes your encoded data to tree.
    TreeNode * deserialize(std::string data)
    {
        if (data.empty()) return nullptr;

        int len = data.size() / sizeof(int);
        auto preorder = reinterpret_cast<int *>(data.data());

        TreeNode * root = new TreeNode(preorder[0]);

        std::stack<TreeNode *> st;
        st.emplace(root);

        for (int i = 1; i < len; ++i)
        {
            TreeNode * curr = st.top();
            TreeNode * child = new TreeNode(preorder[i]);

            while (!st.empty() && st.top()->val < child->val)
            {
                curr = st.top();
                st.pop();
            }

            if (child->val < curr->val) curr->left = child;
            else                        curr->right = child;
            
            st.emplace(child);
        }

        return root;
    }
};

// Your Codec object will be instantiated and called as such:
// Codec* ser = new Codec();
// Codec* deser = new Codec();
// string tree = ser->serialize(root);
// TreeNode* ans = deser->deserialize(tree);
// return ans;