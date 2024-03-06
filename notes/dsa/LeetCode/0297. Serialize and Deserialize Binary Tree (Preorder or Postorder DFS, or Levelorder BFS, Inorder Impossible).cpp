/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Preorder
{
public:
    std::string serialize(TreeNode * root)
    {
        if (!root) return "# ";
        return std::to_string(root->val) + " " + serialize(root->left) + serialize(root->right);
    }

    TreeNode * deserialize(std::string data)
    {
        std::istringstream sin(data);

        std::function<TreeNode * ()> f = [&sin, &f]() -> TreeNode *
        {
            static std::string line;
            if (!std::getline(sin, line, ' ')) return nullptr;
            if (line.front() == '#') return nullptr;
            TreeNode * root = new TreeNode(std::stoi(line));
            root->left = f();
            root->right = f();
            return root;
        };

        return f();
    }
};

class Levelorder
{
public:
    std::string serialize(TreeNode * root)
    {
        if (!root) return "# ";

        std::string ans;

        std::queue<TreeNode *> que;
        que.emplace(root);

        while (!que.empty())
        {
            TreeNode * p = que.front();
            que.pop();
            
            if (!p)
            {
                ans += "# ";
                continue;
            }
            
            ans += std::to_string(p->val) + " ";
            que.emplace(p->left);
            que.emplace(p->right);
        }

        return ans;
    }

    TreeNode * deserialize(std::string data)
    {
        if (data.empty() || data.front() == '#') return nullptr;

        std::istringstream sin(data);
        std::string line;
        std::getline(sin, line, ' ');

        TreeNode * root = new TreeNode(std::stoi(line));

        std::queue<TreeNode *> que;
        que.emplace(root);

        while (!que.empty())
        {
            int levelSize = que.size();

            for (int i = 0; i < levelSize; ++i)
            {
                TreeNode * p = que.front();
                que.pop();

                std::getline(sin, line, ' ');

                if (line.front() != '#') 
                {
                    p->left = new TreeNode(std::stoi(line));
                    que.emplace(p->left);
                }

                std::getline(sin, line, ' ');

                if (line.front() != '#')
                {
                    p->right = new TreeNode(std::stoi(line));
                    que.emplace(p->right);
                }
            }
        }

        return root;
    }
};

// using Codec = Preorder;
using Codec = Levelorder;

// Your Codec object will be instantiated and called as such:
// Codec ser, deser;
// TreeNode* ans = deser.deserialize(ser.serialize(root));