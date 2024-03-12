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
class StringComparasion  // O(m + n) time, O(m + n) space
{
public:
    bool isSubtree(TreeNode * root, TreeNode * subRoot)
    {
        std::string t = serialize(root);
        std::string p = serialize(subRoot);
        int match = kmp(t, p);

        // Latter: Fix for trees [12] vs [2]. 
        return match != -1 && (match == 0 || t[match - 1] == ' ');
    }

private:
    static std::string serialize(TreeNode * root)
    {
        std::string ans;
        serializeImpl(root, ans);
        return ans;
    }

    static void serializeImpl(TreeNode * root, std::string & ans)
    {
        if (!root)
        {
            ans += "# ";
            return;
        }

        ans += std::to_string(root->val) + ' ';
        serializeImpl(root->left, ans);
        serializeImpl(root->right, ans);
    }

    static std::vector<int> buildProperPrefix(const std::string & s)
    {
        auto n = static_cast<const int>(s.size());   
        std::vector<int> ppx(n, 0);

        for (int t = 0, j = 1; j < n; ++j)
        {
            while (0 < t && s[t] != s[j]) t = ppx[t - 1];
            t = ppx[j] = t + (s[t] == s[j]);
        }

        return ppx;
    }

    static int kmp(const std::string & t, const std::string & p)
    {
        auto m = static_cast<const int>(t.size());
        auto n = static_cast<const int>(p.size());

        std::vector<int> ppx = buildProperPrefix(p);

        int i = 0;
        int j = 0;

        while (i < m && j < n)
        {
            if (t[i] == p[j]) ++i, ++j;
            else if (j == 0) ++i;
            else j = ppx[j - 1];
        }

        return j == n ? i - j : -1;
    }
};

class Solution  // O(mn) time, O(m + n) space
{
public:
    bool isSubtree(TreeNode * root, TreeNode * subRoot)
    {
        return compare(root, subRoot);
    }

private:
    static bool compare(TreeNode * p, TreeNode * q)
    {
        if (!p || !q) return !p && !q;   
        return isSameTree(p, q) || compare(p->left, q) || compare(p->right, q);
    }

    static bool isSameTree(TreeNode * p, TreeNode * q)
    {
        if (!p || !q) return !p && !q;

        return p->val == q->val && 
               isSameTree(p->left, q->left) && 
               isSameTree(p->right, q->right);
    }
};
