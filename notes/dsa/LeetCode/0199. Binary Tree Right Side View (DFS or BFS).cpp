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
    std::vector<int> rightSideView(TreeNode * root)
    {
        if (!root) return {};
        std::vector<int> ans;

        // // Recursive DFS.
        // dfs(root, 0, ans);

        // BFS. 
        // Always enqueue left children first, 
        // then the rightmost node on current level 
        // will be enqueued as the last one. 
        std::queue<TreeNode *> qu;
        qu.emplace(root);

        while (!qu.empty())
        {
            int depthLength = qu.size();

            for (int i = 0; i < depthLength; ++i)
            {
                TreeNode * curr = qu.front();
                qu.pop();
                if (i == depthLength - 1) ans.emplace_back(curr->val);
                if (curr->left) qu.emplace(curr->left);
                if (curr->right) qu.emplace(curr->right);
            }
        }

        return ans;
    }

private:
    static void dfs(TreeNode * curr, int depth, std::vector<int> & ans)
    {
        if (depth == ans.size()) ans.emplace_back(curr->val);
        if (curr->right) dfs(curr->right, depth + 1, ans);
        if (curr->left) dfs(curr->left, depth + 1, ans);
    }
};