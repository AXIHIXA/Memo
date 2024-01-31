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
class BSTIterator
{
public:
    BSTIterator(TreeNode * root) : curr(root)
    {
        
    }
    
    int next()
    {
        while (!st.empty() || curr)
        {
            if (curr)
            {
                st.emplace(curr);
                curr = curr->left;
            }
            else
            {
                curr = st.top();
                st.pop();
                int v = curr->val;
                curr = curr->right;
                return v;
            }
        }

        return -1;
    }
    
    bool hasNext()
    {
        return !st.empty() || curr;
    }

private:
    std::stack<TreeNode *> st;
    TreeNode * curr = nullptr;
};

/**
 * Your BSTIterator object will be instantiated and called as such:
 * BSTIterator* obj = new BSTIterator(root);
 * int param_1 = obj->next();
 * bool param_2 = obj->hasNext();
 */