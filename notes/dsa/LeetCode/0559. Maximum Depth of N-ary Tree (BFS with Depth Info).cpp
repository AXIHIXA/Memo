/*
// Definition for a Node.
class Node {
public:
    int val;
    vector<Node*> children;

    Node() {}

    Node(int _val) {
        val = _val;
    }

    Node(int _val, vector<Node*> _children) {
        val = _val;
        children = _children;
    }
};
*/

class Solution
{
public:
    int maxDepth(Node * root)
    {
        if (!root) return 0;

        std::queue<std::pair<Node *, int>> que;
        que.emplace(root, 1);

        int ans = 0;

        while (!que.empty())
        {
            auto [p, d] = que.front();
            que.pop();

            ans = std::max(ans, d);

            for (Node * c : p->children)
            {
                que.emplace(c, d + 1);
            }
        }

        return ans;
    }
};