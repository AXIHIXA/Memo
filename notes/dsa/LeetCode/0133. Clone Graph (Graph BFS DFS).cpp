/*
// Definition for a Node.
class Node {
public:
    int val;
    vector<Node*> neighbors;
    Node() {
        val = 0;
        neighbors = vector<Node*>();
    }
    Node(int _val) {
        val = _val;
        neighbors = vector<Node*>();
    }
    Node(int _val, vector<Node*> _neighbors) {
        val = _val;
        neighbors = _neighbors;
    }
};
*/

int init = []
{
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);
    std::cout.tie(nullptr);
    return 0;
}();

class Solution
{
public:
    Node * cloneGraph(Node * node)
    {
        if (!node) return nullptr;
        
        Node * ans = new Node(node->val);
        std::unordered_map<Node *, Node *> hm {{node, ans}};

        std::queue<Node *> qu;
        qu.emplace(node);

        while (!qu.empty())
        {
            Node * curr = qu.front();
            qu.pop();

            Node * cloned = hm.at(curr);
            
            for (Node * neighbor : curr->neighbors)
            {
                auto it = hm.find(neighbor);
                
                if (it == hm.end())
                {
                    qu.emplace(neighbor);
                    it = hm.emplace(neighbor, new Node(neighbor->val)).first;
                }
                
                cloned->neighbors.emplace_back(it->second);
            }
        }

        return ans;
    }
};