//
// Created by AX on 2021/4/22.
//

#ifndef CPPDEMO_S35_H
#define CPPDEMO_S35_H


struct S35
{
    S35() { printf("S35::S35()\n"); }
    explicit S35(const int i) : p(new int(i)) { printf("S35::S35(const int &)\n"); }
    S35(const S35 & rhs) : p(new int(*rhs.p)) { printf("S35::S35(const S35 &)\n"); }
    S35(S35 && rhs) noexcept : p(std::move(rhs.p)) { printf("S35::S35(S35 &&)\n"); }
    virtual ~S35() { printf("S35::~S35()\n"); };

    S35 & operator=(const S35 & rhs)
    {
        printf("S35::operator=(const S35 &)\n");
        if (this != &rhs) p = std::make_unique<int>(*rhs.p);
        return *this;
    }

    S35 & operator=(S35 && rhs) noexcept
    {
        printf("S35::operator=(S35 &&)\n");
        if (this != &rhs) p = std::move(rhs.p);
        return *this;
    }

    //    // copy-and-swap assign operator deals with self-assignment
    //    // and servers automatically as both copy and move assign operator
    //    S35 & operator=(S35 rhs)
    //    {
    //        printf("S35::operator=(S35)\n");
    //        using std::swap;
    //        swap(p, rhs.p);
    //        return *this;
    //    }

    // when used as condition, this explicit operator will still be applied by compiler implicitly
    // "this is a feature, NOT a bug. " -- Microsoft
    explicit operator bool() const { return static_cast<bool>(*p); }

    std::unique_ptr<int> p{new int(0)};
};


#endif  // CPPDEMO_S35_H
