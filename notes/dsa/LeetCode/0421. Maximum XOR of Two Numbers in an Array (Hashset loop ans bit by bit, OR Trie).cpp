class Trie
{
public:
    int findMaximumXOR(std::vector<int> & nums)
    {
        int maxi = *std::max_element(nums.cbegin(), nums.cend());
        bitLength = 0;
        for ( ; maxi; maxi >>= 1) ++bitLength;
        clear();

        int ans = 0;

        for (int x : nums)
        {
            insert(x);
            ans = std::max(ans, findMaxXor(x));
        }

        return ans;
    }

private:
    static void clear()
    {
        cnt = 1;
        
        for (auto & node : tree) 
        {
            std::fill(node.begin(), node.end(), 0);
        }
    }

    static void insert(int x)
    {
        int curr = 1;

        for (int i = bitLength, path; 0 <= i; --i)
        {
            path = (x >> i) & 1;
            if (tree[curr][path] == 0) tree[curr][path] = ++cnt;
            curr = tree[curr][path];
        }
    }

    static int findMaxXor(int x)
    {
        int ans = 0;
        int curr = 1;

        for (int i = bitLength, path; 0 <= i; --i)
        {
            path = ((x >> i) & 1) ^ 1;

            if (tree[curr][path] != 0)
            {
                ans |= (1 << i);
                curr = tree[curr][path];
            }
            else
            {
                curr = tree[curr][path ^ 1];
            }
        }

        return ans;
    }

private:
    // Number of words == 2e5
    // Number of chars == 2e5 * 32 (max bit length)
    // Number of nodes needed == 2e5 * 32 * 2 (0 or 1) == 12'800'000
    static constexpr int kMaxSize = 12'800'000;

    static int cnt;
    static int bitLength;
    static std::array<std::array<int, 2>, kMaxSize> tree;
};

int Trie::cnt;
int Trie::bitLength;
std::array<std::array<int, 2>, Trie::kMaxSize> Trie::tree;

class HashSet
{
public:
    int findMaximumXOR(std::vector<int> & nums)
    {
        int maxi = *std::max_element(nums.cbegin(), nums.cend());
        int bitLength = 0;
        for ( ; maxi; maxi >>= 1) ++bitLength;

        int ans = 0;
        int mask = 0;
        std::unordered_set<int> hashSet;

        for (int i = bitLength; 0 <= i; --i)
        {
            mask |= (1 << i);
            int next = ans | (1 << i);
            hashSet.clear();

            for (int x : nums)
            {
                x &= mask;

                if (hashSet.contains(next ^ x))
                {
                    ans = next;
                    break;
                }

                hashSet.insert(x);
            }
        }

        return ans;
    }
};

// using Solution = Trie;
using Solution = HashSet;
