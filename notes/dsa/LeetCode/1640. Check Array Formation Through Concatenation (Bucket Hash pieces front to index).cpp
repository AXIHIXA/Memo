class Solution
{
public:
    bool canFormArray(std::vector<int> & arr, std::vector<std::vector<int>> & pieces)
    {
        std::vector<int> bucket(101, -1);

        for (int i = 0; i < pieces.size(); ++i)
        {
            bucket[pieces[i][0]] = i;
        }

        for (int i = 0; i < arr.size(); )
        {
            if (bucket[arr[i]] == -1)
            {
                return false;
            }

            const auto & p = pieces[bucket[arr[i]]];

            for (int j = 0; j < p.size(); ++j, ++i)
            {
                if (arr[i] != p[j])
                {
                    return false;
                }
            }
        }

        return true;
    }
};