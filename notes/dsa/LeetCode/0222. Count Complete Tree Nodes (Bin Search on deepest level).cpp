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
    int countNodes(TreeNode * root)
    {
        if (!root) return 0;

        int d = depth(root);
        if (d == 0) return 1;

        int ll = 1, rr = (1 << d) - 1;

        while (ll <= rr)
        {
            int mi = ll + ((rr - ll) >> 1);
            if (exists(mi, d, root)) ll = mi + 1;
            else                     rr = mi - 1;
        }

        return (1 << d) - 1 + ll;
    }

private:
    static int depth(TreeNode * root)
    {
        int ans = 0;
        while (root->left)
        {
            root = root->left;
            ++ans;
        }
        return ans;
    }

    static bool exists(int idx, int d, TreeNode * node)
    {
        int ll = 0, rr = (1 << d) - 1;

        for (int i = 0; i < d; ++i)
        {
            int mi = ll + ((rr - ll) >> 1);

            if (idx <= mi)
            {
                node = node->left;
                rr = mi;
            }
            else
            {
                node = node->right;
                ll = mi + 1;
            }
        }

        return node;
    }
};