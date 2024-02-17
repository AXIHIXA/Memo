class SnapshotArray
{
public:
    SnapshotArray(int length)
    {
        data.resize(length, {{0, 0}});
    }
    
    void set(int index, int val)
    {
        data[index].emplace_back(currentSnap, val);
    }
    
    int snap()
    {
        return currentSnap++;
    }
    
    int get(int index, int snap_id)
    {
        auto it = std::upper_bound(
                data[index].cbegin(), 
                data[index].cend(), 
                std::make_pair(snap_id, std::numeric_limits<int>::max())
        );
        
        return std::prev(it)->second;
    }

private:
    // data[i]: vector of (snap id, value of index i at this snap). 
    std::vector<std::vector<std::pair<int, int>>> data;
    int currentSnap = 0;
};

/**
 * Your SnapshotArray object will be instantiated and called as such:
 * SnapshotArray* obj = new SnapshotArray(length);
 * obj->set(index,val);
 * int param_2 = obj->snap();
 * int param_3 = obj->get(index,snap_id);
 */