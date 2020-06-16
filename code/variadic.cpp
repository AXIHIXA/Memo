#include <boost/core/demangle.hpp>
#include <bits/stdc++.h>


// variadic template class expansion
// via partial specialization
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


// variadic template class expansion
// via inheritance
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


// variadic template function expansion
// via recursion
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


// variadic template function expansion
// via initializer list of comma expressions
template <typename ... Args>
void pf2(Args && ... args)
{
    int a[] = {(std::cout << args << ", ", 0) ...};
    std::cout << std::endl;
}


// a fake "variadic variable" function
// can be done by passing in an initializer list
template <typename T>
void pf3(std::initializer_list<T> il)
{
    for (const T & item : il)
    {
        std::cout << item << ", ";
    }

    std::cout << std::endl;
}


// a test on perfect forwarding of variadic template functions

// variadic template function expansion
// via recursion
template <typename T>
void fun4(T && t)
{
    std::cout << "void fun3(T && t) " << t << std::endl;
}

void fun4(std::string && s)
{
    std::cout << "void fun3(std::string && s) " << s << std::endl;
}

template <typename T, typename ... Args>
void fun4(T && t, Args && ... args)
{
    std::cout << "void fun4(T && t, Args && ... args) " << t << std::endl;
    fun4(std::forward<Args>(args) ...);
}

// this one does perfect forwarding, successfully calling the std::string && specialization 
template <typename ... Args>
void fun3_1(Args && ... args)
{
    fun4(std::forward<Args>(args) ...);
}

// this one doesn't do perfect forwarding, only calling the template
template <typename ... Args>
void fun3_2(Args && ... args)
{
    fun4(args ...);
}


int main(int argc, char * argv[])
{
    std::cout << Sum<char, short, int, long long>::value << std::endl;

    std::cout << boost::core::demangle(typeid(MakeIndexes<4>::type).name()) << std::endl;

    pf1(0, 1, 2, 3, 4);

    pf2(0, 1, 2, 3, 4);

    pf3({0, 1, 2, 3, 4});
    
    fun3_1(1, std::string {"rval"});  // void fun4(T && t, Args && ... args) 1
                                      // void fun3(std::string && s) rval
                              
    fun3_2(1, std::string {"rval"});  // void fun4(T && t, Args && ... args) 1
                                      // void fun3(T && t) rval

    return EXIT_SUCCESS;
}
