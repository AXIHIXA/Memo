/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Codec 
{
public:

    // Encodes a tree to a single string.
    string serialize(TreeNode * root) 
    {
        if (!root) return "#\n";
        return to_string(root->val) + "\n" + serialize(root->left) + serialize(root->right);
    }

    // Decodes your encoded data to tree.
    TreeNode * deserialize(string data) 
    {
        if (data[0] == '#') return nullptr;
        stringstream ss(data);
        return helper(ss);
    }

private:
    TreeNode * helper(stringstream & ss)
    {
        string line;
        getline(ss, line);
        
        if (line[0] == '#') return nullptr;

        TreeNode * root = new TreeNode(stoi(line));
        root->left = helper(ss);
        root->right = helper(ss);

        return root;
    }

    void dfs(TreeNode * root)
    {
        if (!root)
        {
            return;
        }
        
        visit(root);
        dfs(root->left);
        dfs(root->right);
    }

    void bfs(TreeNode * root)
    {
        queue<TreeNode *> q;

        q.push(root);

        while (!q.empty())
        {
            TreeNode * p = q.front();
            q.pop();

            if (p)
            {
                q.push(p->left);
                q.push(p->right);
            }

            visit(p);
        }
    }

    void visit(TreeNode * node)
    {
        if (!node)
        {
            cout << "n\n";
            return;
        }

        cout << node->val << '\n';
    }
};

// Your Codec object will be instantiated and called as such:
// Codec ser, deser;
// TreeNode* ans = deser.deserialize(ser.serialize(root));
