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
        
        int levels = 0;
        int bottomMaxSize = 1;

        for (TreeNode * p = root->left; p; p = p->left)
        {
            ++levels;
            bottomMaxSize <<= 1;
        }

        auto exists = [root, levels](int k) -> bool
        {
            int i = levels - 1;
            TreeNode * p = root;
            
            for ( ; 0 <= i && p != nullptr; --i)
            {
                if ((k >> i) & 1)
                {
                    p = p->right;
                }
                else
                {
                    p = p->left;
                }
            }

            return i == -1 && p != nullptr;
        };

        int lo = 0;
        int hi = bottomMaxSize;

        while (lo < hi)
        {
            int mi = lo + ((hi - lo) >> 1);
            if (exists(mi)) lo = mi + 1;
            else hi = mi;
        }

        return bottomMaxSize - 1 + lo;
    }
};