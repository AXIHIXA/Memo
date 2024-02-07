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
    ListNode * sortList(ListNode * head)
    {
        if (!head || !head->next) return head;

        #ifdef DEBUG
            std::cout << "===========================\nsort [ ";
            for (ListNode * pp = head; pp; pp = pp->next)
                std::cout << pp->val << ' ';
            std::cout << "]\n";
        #endif

        ListNode pivotNode(0);
        ListNode * pivot = &pivotNode;
        pivot->next = head;

        int listLen = 0;
        while (head) { head = head->next; ++listLen; }
        head = pivot;

        for (int step = 1; step < listLen; step <<= 1)
        {
            #ifdef DEBUG
                std::cout << "step = " << step << ":\n";
            #endif

            while (head && head->next)
            {
                auto [h2, h3] = split(head->next, step);
                // head [head->next ... nullptr] [h2 ... nullptr] h3

                auto [ll, rr] = merge(head->next, h2);
                // head [ll ... rr] h3

                head->next = ll;
                rr->next = h3;
                head = rr;
            }
            
            #ifdef DEBUG
                for (head = pivot->next; head; head = head->next)
                    std::cout << head->val << ' ';
                std::cout << '\n';
            #endif

            head = pivot;
        }

        return pivot->next;
    }

private:
    // @return: pair {h2, h3}
    //          [head ... nullptr] of length size,
    //          [h2 ... nullptr] h3 of length <= size. 
    static std::pair<ListNode *, ListNode *> split(ListNode * head, int size)
    {
        #ifdef DEBUG
            std::cout << "split [ ";
            ListNode * pp = head;
            for (int i = 0; i < 2 * size && pp; ++i, pp = pp->next)
                std::cout << pp->val << ' ';
            std::cout << "] ";
            pp ? std::cout << pp->val : std::cout << "nullptr";
            std::cout << " into [ ";
        #endif

        ListNode pivotNode(0);
        ListNode * pivot = &pivotNode;
        pivot->next = head;
        
        ListNode * slow = pivot;
        ListNode * fast = pivot;

        for (int i = 0; i < size && (slow->next || fast->next); ++i)
        {
            if (fast->next) fast = fast->next->next ? fast->next->next : fast->next;
            if (slow->next) slow = slow->next;
        }

        // pivot [head slow] [slow->next ... fast] fast->next

        ListNode * slowNext = slow->next;
        slow->next = nullptr;

        ListNode * fastNext = fast->next;
        fast->next = nullptr;
        
        #ifdef DEBUG
            for (ListNode * pp = head; pp; pp = pp->next)
                std::cout << pp->val << ' ';
            std::cout << " ] and [ ";
            for (ListNode * pp = slowNext; pp; pp = pp->next)
                std::cout << pp->val << ' ';
            std::cout << "] "; 
            fastNext ? std::cout << fastNext->val : std::cout << "nullptr";
            std::cout << "\n";
        #endif

        return {slowNext, fastNext};
    }

    static std::pair<ListNode *, ListNode *> merge(ListNode * h1, ListNode * h2)
    {
        #ifdef DEBUG
            std::cout << "merge [ ";
            for (ListNode * p = h1; p; p = p->next) 
                std::cout << p->val << ' ';
            std::cout << "] and [ ";
            for (ListNode * p = h2; p; p = p->next) 
                std::cout << p->val << ' ';
            std::cout << "] into [ ";
        #endif
        
        ListNode pivotNode = ListNode(0);
        ListNode * pivot = &pivotNode;
        ListNode * p = pivot;

        while (h1 && h2)
        {
            if (h1 && (!h2 || h1->val <= h2->val)) { p->next = h1; p = p->next; h1 = h1->next; }
            if (h2 && (!h1 || h2->val <  h1->val)) { p->next = h2; p = p->next; h2 = h2->next; }
        }

        if (h1) p->next = h1;
        if (h2) p->next = h2;

        while (p->next) p = p->next;

        #ifdef DEBUG
            for (ListNode * pp = pivot->next; pp != p; pp = pp->next)
                std::cout << pp->val << ' ';
            std::cout << p->val << " ]\n";
        #endif

        return {pivot->next, p};
    }
};