/*
// Definition for a QuadTree node.
class Node {
public:
    bool val;
    bool isLeaf;
    Node* topLeft;
    Node* topRight;
    Node* bottomLeft;
    Node* bottomRight;
    
    Node() {
        val = false;
        isLeaf = false;
        topLeft = NULL;
        topRight = NULL;
        bottomLeft = NULL;
        bottomRight = NULL;
    }
    
    Node(bool _val, bool _isLeaf) {
        val = _val;
        isLeaf = _isLeaf;
        topLeft = NULL;
        topRight = NULL;
        bottomLeft = NULL;
        bottomRight = NULL;
    }
    
    Node(bool _val, bool _isLeaf, Node* _topLeft, Node* _topRight, Node* _bottomLeft, Node* _bottomRight) {
        val = _val;
        isLeaf = _isLeaf;
        topLeft = _topLeft;
        topRight = _topRight;
        bottomLeft = _bottomLeft;
        bottomRight = _bottomRight;
    }
};
*/

class Solution
{
public:
    Node * construct(std::vector<std::vector<int>> & grid)
    {
        return build(grid, 0, 0, grid.size());
    }

private:
    static Node * build(const std::vector<std::vector<int>> & grid, int r, int c, int n)
    {
        if (n == 1) return new Node(grid[r][c], true);

        int half = n >> 1;
        Node * topLeft = build(grid, r, c, half);
        Node * topRight = build(grid, r, c + half, half);
        Node * bottomLeft = build(grid, r + half, c, half);
        Node * bottomRight = build(grid, r + half, c + half, half);

        if (
                topLeft->isLeaf && topRight->isLeaf && bottomLeft->isLeaf && bottomRight->isLeaf && 
                topLeft->val == topRight->val && 
                topRight->val == bottomLeft->val && 
                bottomLeft->val == bottomRight->val
        )
        {
            int val = topLeft->val;
            delete topLeft;
            delete topRight;
            delete bottomLeft;
            delete bottomRight;
            return new Node(val, true);
        }

        return new Node(false, false, topLeft, topRight, bottomLeft, bottomRight);
    }
};