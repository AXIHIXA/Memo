/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Recursion
{
public:
    TreeNode * lowestCommonAncestor(TreeNode * root, TreeNode * p, TreeNode * q)
    {
        if (!root || root == p || root == q) return root;

        TreeNode * left = lowestCommonAncestor(root->left, p, q);
        TreeNode * right = lowestCommonAncestor(root->right, p, q);
        
        if (left && right) return root;
        if (!left && !right) return nullptr;

        return left ? left : right;
    }
};

class Iteration
{
public:
    TreeNode * lowestCommonAncestor(TreeNode * root, TreeNode * p, TreeNode * q)
    {
        std::stack<std::pair<TreeNode *, int>> stk;
        stk.emplace(root, 0);

        int targetNodesFound = 0;
        TreeNode * ans = nullptr;

        while (!stk.empty())
        {
            auto & [node, childrenVisited] = stk.top();

            if (childrenVisited < 2)
            {
                if (childrenVisited == 0)
                {
                    if (node == p || node == q)
                    {
                        ++targetNodesFound;

                        if (targetNodesFound == 2)
                        {
                            return ans;
                        }

                        ans = node;
                    }

                    if (node->left)
                    {
                        stk.emplace(node->left, 0);
                    }
                }
                else
                {
                    if (node->right)
                    {
                        stk.emplace(node->right, 0);
                    }
                }

                ++childrenVisited;
            }
            else
            {
                stk.pop();

                if (ans == node && !stk.empty())
                {
                    ans = stk.top().first;
                }
            }
        }

        return nullptr;
    }
};

// using Solution = Recursion;
using Solution = Iteration;
