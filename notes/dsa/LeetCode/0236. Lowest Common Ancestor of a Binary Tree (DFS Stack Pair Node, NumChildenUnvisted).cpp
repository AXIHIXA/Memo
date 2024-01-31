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
        // Parent node, Number of children unvisited. 
        std::stack<std::pair<TreeNode *, int>> st;
        st.emplace(root, 2);
        bool foundOneNode = false;
        TreeNode * lca = nullptr;

        while (!st.empty())
        {
            auto & [curr, numChildrenUnvisited] = st.top();

            if (numChildrenUnvisited)
            {
                TreeNode * child = nullptr;
                
                if (numChildrenUnvisited == 2)
                {
                    if (curr == p || curr == q)
                    {
                        if (!foundOneNode) foundOneNode = true;
                        else               return lca;

                        lca = curr;
                    }
                    
                    child = curr->left;
                }
                else
                {
                    child = curr->right;
                }

                --numChildrenUnvisited;
                if (child) st.emplace(child, 2);
            }
            else
            {
                st.pop();
                if (lca == curr && foundOneNode) lca = st.top().first;
            }
        }

        return nullptr;
    }
};