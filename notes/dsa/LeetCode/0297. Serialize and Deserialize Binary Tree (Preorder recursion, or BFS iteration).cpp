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
    std::string serialize(TreeNode * root) 
    {
        std::queue<TreeNode *> qu;
        qu.emplace(root);

        std::string ans;

        while (!qu.empty())
        {
            TreeNode * curr = qu.front();
            qu.pop();

            if (!curr)
            {
                ans += "# ";
                continue;
            }

            ans += std::to_string(curr->val) + " ";
            qu.emplace(curr->left);
            qu.emplace(curr->right);
        }

        return ans;
    }

    // Decodes your encoded data to tree.
    TreeNode * deserialize(std::string data) 
    {
        if (data[0] == '#') return nullptr;

        std::stringstream ss(data);
        std::string line;
        std::getline(ss, line, ' ');

        TreeNode * root = new TreeNode(std::stoi(line));
        
        std::queue<TreeNode *> qu;
        qu.emplace(root);

        while (!qu.empty())
        {
            TreeNode * curr = qu.front();
            qu.pop();

            std::getline(ss, line, ' ');
            TreeNode * left = line[0] == '#' ? nullptr : new TreeNode(std::stoi(line));
            curr->left = left;
            if (left) qu.emplace(left);
            
            std::getline(ss, line, ' ');
            TreeNode * right = line[0] == '#' ? nullptr : new TreeNode(std::stoi(line));
            curr->right = right;
            if (right) qu.emplace(right);
        }

        return root;
    }

private:
    static TreeNode * build(std::stringstream & ss)
    {
        std::string line; 
        std::getline(ss, line, ' ');
        if (line == "#") return nullptr;
        TreeNode * root = new TreeNode(std::stoi(line));
        root->left = build(ss);
        root->right = build(ss);
        return root;
    }
};

class CodecPreorder
{
public:
    // Encodes a tree to a single string.
    std::string serialize(TreeNode * root) 
    {
        if (!root) return "# ";
        return std::to_string(root->val) + " " + serialize(root->left) + serialize(root->right);
    }

    // Decodes your encoded data to tree.
    TreeNode * deserialize(std::string data) 
    {
        std::stringstream ss(data);
        return build(ss);
    }

private:
    static TreeNode * build(std::stringstream & ss)
    {
        std::string line; 
        std::getline(ss, line, ' ');
        if (line == "#") return nullptr;
        TreeNode * root = new TreeNode(std::stoi(line));
        root->left = build(ss);
        root->right = build(ss);
        return root;
    }
};

// Your Codec object will be instantiated and called as such:
// Codec ser, deser;
// TreeNode* ans = deser.deserialize(ser.serialize(root));
