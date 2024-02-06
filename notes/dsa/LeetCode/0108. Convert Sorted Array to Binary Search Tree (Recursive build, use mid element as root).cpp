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
    TreeNode * sortedArrayToBST(std::vector<int> & nums)
    {
        return build(nums, 0, nums.size());
    }

private:    
    static TreeNode * build(
            const std::vector<int> & nums, 
            int lo, 
            int hi
    )
    {
        if (hi < lo + 1) return nullptr;
        int mi = lo + ((hi - lo) >> 1);
        TreeNode * root = new TreeNode(nums[mi]);
        root->left = build(nums, lo, mi);
        root->right = build(nums, mi + 1, hi);
        return root;
    }
};