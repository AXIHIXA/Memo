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
    bool isSameTree(TreeNode * p, TreeNode * q)
    {
        if (!p) return !q;
        
        std::stack<TreeNode *> st1;
        std::stack<TreeNode *> st2;

        // In-Order Traversal. 
        while (!st1.empty() || p)
        {
            if (p)
            {
                st1.emplace(p);
                p = p->left;

                if (!q) return false;
                st2.emplace(q);
                q = q->left;
                
                // Avoid testcase #60, p = [1], q = [1,null,2], 
                // i.e., p is done traversal but q has more nodes unvisited. 
                if ((p == nullptr) != (q == nullptr)) return false;
            }
            else
            {
                p = st1.top();
                st1.pop();

                if (st2.empty()) return false;
                q = st2.top();
                st2.pop();

                if (p->val != q->val) return false;

                p = p->right;
                q = q->right;

                // Avoid testcase #60, p = [1], q = [1,null,2], 
                // i.e., p is done traversal but q has more nodes unvisited. 
                if ((p == nullptr) != (q == nullptr)) return false;
            }
        }

        return true;
    }
};