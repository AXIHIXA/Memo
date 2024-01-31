#include <iostream>
#include <stack>
#include <vector>


struct Node
{
    Node() = default;
    Node(int v, Node * l, Node * r) : val(v), left(l), right(r) {}

    int val {0};
    Node * left {nullptr};
    Node * right {nullptr};
};


void preorderTraverse(Node * root)
{
    if (!root) return;
    std::stack<Node *> st;
    st.emplace(root);
    while (!st.empty())
    {
        Node * curr = st.top();
        st.pop();
        std::cout << curr->val << ' ';
        if (curr->right) st.emplace(curr->right);
        if (curr->left) st.emplace(curr->left);
    }
    std::cout << '\n';
}


void inorderTraverse(Node * root)
{
    if (!root) return;
    std::stack<Node *> st;
    while (!st.empty() || root)
    {
        if (root)
        {
            st.emplace(root);
            root = root->left;
        }
        else
        {
            root = st.top();
            st.pop();
            std::cout << root->val << ' ';
            root = root->right;
        }
    }
    std::cout << '\n';
}


void postorderTraverse(Node * root)
{
    if (!root) return;
    std::stack<Node *> st;
    st.emplace(root);
    while (!st.empty())
    {
        Node * curr = st.top();
        if (curr->left && curr->left != root && curr->right != root)
        {
            st.emplace(curr->left);
        }
        else if (curr->right && curr->right != root)
        {
            st.emplace(curr->right);
        }
        else
        {
            root = curr;
            std::cout << curr->val << ' ';
            st.pop();
        }
    }
    std::cout << '\n';
}


void preorderTraverseMorris(Node * root)
{
    while (root)
    {
        if (!root->left)
        {
            std::cout << root->val << ' ';
            root = root->right;
        }
        else
        {
            Node * prev = root->left;
            while (prev->right && prev->right != root) prev = prev->right;

            if (!prev->right)
            {
                std::cout << root->val << ' ';
                prev->right = root;
                root = root->left;
            }
            else
            {
                prev->right = nullptr;
                root = root->right;
            }
        }
    }
    std::cout << '\n';
}


void inorderTraverseMorris(Node * root)
{
    while (root)
    {
        if (!root->left)
        {
            std::cout << root->val << ' ';
            root = root->right;
        }
        else
        {
            Node * prev = root->left;
            while (prev->right && prev->right != root) prev = prev->right;

            if (!prev->right)
            {
                prev->right = root;
                root = root->left;
            }
            else
            {
                std::cout << root->val << ' ';
                prev->right = nullptr;
                root = root->right;
            }
        }
    }
    std::cout << '\n';
}


int main(int argc, char * argv[])
{
    //           0
    //      1         2
    //  3      4    5     6
    //    7  8    9  10      
    std::vector<Node> buf(11);
    buf[0] = {0, &buf[1], &buf[2]};
    buf[1] = {1, &buf[3], &buf[4]};
    buf[2] = {2, &buf[5], &buf[6]};
    buf[3] = {3, nullptr, &buf[7]};
    buf[4] = {4, &buf[8], nullptr};
    buf[5] = {5, &buf[9], &buf[10]};
    buf[6] = {6, nullptr, nullptr};
    buf[7] = {7, nullptr, nullptr};
    buf[8] = {8, nullptr, nullptr};
    buf[9] = {9, nullptr, nullptr};
    buf[10] = {10, nullptr, nullptr};
    Node * root = &buf[0];

    preorderTraverse(root);
    preorderTraverseMorris(root);

    inorderTraverse(root);
    inorderTraverseMorris(root);

    postorderTraverse(root);

    return EXIT_SUCCESS;
}
