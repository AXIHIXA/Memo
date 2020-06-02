#include <boost/core/demangle.hpp>
#include <bits/stdc++.h>


template <typename T>
struct Sum<T>
{
    enum
    {
        value = sizeof(T)
    };
};


template <typename T, typename ... Args>
struct Sum
{
    enum
    {
        value = Sum<T>::value + Sum<Args ...>::value
    };
};


template <int ...>
struct IndexSeq
{

};


template <int N, int ... Indexes>
struct MakeIndexes : MakeIndexes<N - 1, N - 1, Indexes ...>
{
};


template <int ... Indexes>
struct MakeIndexes<0, Indexes ...>
{
    typedef IndexSeq<Indexes ...> type;
};


template<typename T>
void pf1(T && t)
{
    std::cout << t << std::endl;
}


template<typename T, typename ... Args>
void pf1(T && t, Args && ... args)
{
    std::cout << t << ", ";
    pf1(std::forward<T>(args) ...);
}


template <typename ... Args>
void pf2(Args && ... args)
{
    int a[] = {(std::cout << args << ", ", 0) ...};
    std::cout << std::endl;
}


template <typename T>
void pf3(std::initializer_list<T> il)
{
    for (const T & item : il)
    {
        std::cout << item << ", ";
    }

    std::cout << std::endl;
}


int main(int argc, char * argv[])
{
    std::cout << Sum<char, short, int, long long>::value << std::endl;

    std::cout << boost::core::demangle(typeid(MakeIndexes<4>::type).name()) << std::endl;

    pf1(0, 1, 2, 3, 4);

    pf2(0, 1, 2, 3, 4);

    pf3({0, 1, 2, 3, 4});

    return EXIT_SUCCESS;
}
