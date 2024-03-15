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
    TreeNode * deleteNode(TreeNode * root, int key)
    {
        if (!root)
        {
            return nullptr;
        }

        if (root->val < key)
        {
            root->right = deleteNode(root->right, key);
        }
        else if (key < root->val)
        {
            root->left = deleteNode(root->left, key);
        }
        else
        {
            if (!root->left && !root->right)
            {
                // delete root;
                root = nullptr;
            }
            else if (root->left)
            {
                root->val = prev(root)->val;
                root->left = deleteNode(root->left, root->val);
            }
            else
            {
                root->val = next(root)->val;
                root->right = deleteNode(root->right, root->val);
            }
        }

        return root;
    }

private:
    static TreeNode * next(TreeNode * root)
    {
        // if (!root || !root->right)
        // {
        //     return nullptr;
        // }

        root = root->right;

        while (root->left)
        {
            root = root->left;
        }

        return root;
    }

    static TreeNode * prev(TreeNode * root)
    {
        // if (!root || !root->left)
        // {
        //     return nullptr;
        // }

        root = root->left;

        while (root->right)
        {
            root = root->right;
        }

        return root;
    }
};