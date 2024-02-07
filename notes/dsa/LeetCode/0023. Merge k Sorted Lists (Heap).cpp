/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution
{
public:
    ListNode * mergeKLists(std::vector<ListNode *> & lists)
    {
        if (lists.empty()) return nullptr;

        std::priority_queue<ListNode *, std::vector<ListNode *>, Compare> heap;
        for (ListNode * p : lists) if (p) heap.push(p);

        ListNode pivot;
        ListNode * head = &pivot;

        while (!heap.empty())
        {
            ListNode * curr = heap.top();
            heap.pop();

            head->next = curr;
            head = curr;

            if (curr->next) heap.push(curr->next);
        }

        return pivot.next;
    }

private:
    struct Compare
    {
        bool operator()(ListNode * a, ListNode * b)
        {
            return a->val > b->val;
        }
    };
};