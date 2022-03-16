#ifndef ST_DEBUG_H
#define ST_DEBUG_H


#include <cstdio>


#ifdef ST_COLOR
#define KNRM "\033[0m"
#define KRED "\033[1;31m"
#define KGRN "\033[1;32m"
#define KYEL "\033[1;33m"
#define KBLU "\033[1;34m"
#define KMAG "\033[1;35m"
#define KCYN "\033[1;36m"
#define KWHT "\033[1;37m"
#define KBWN "\033[0;33m"
#else
#define KNRM ""
#define KRED ""
#define KGRN ""
#define KYEL ""
#define KBLU ""
#define KMAG ""
#define KCYN ""
#define KWHT ""
#define KBWN ""
#endif


#ifdef ST_VERBOSE
#define ST_DEBUG
#define ST_INFO
#define ST_WARN
#define ST_ERROR
#define ST_SUCCESS
#endif


#ifdef ST_DEBUG
#define st_debug(S, ...)                                                           \
    do                                                                             \
    {                                                                              \
        fprintf(stdout, KMAG "[DEBUG] %s @ %s:%d " KNRM S "\n",                    \
                __extension__ __func__, __FILE__, __LINE__, ##__VA_ARGS__);        \
    }                                                                              \
    while (0)
#define st_mag(S, ...)                                                             \
    do                                                                             \
    {                                                                              \
        fprintf(stdout, KMAG "%s @ %s:%d " KNRM S "\n",                            \
                __extension__ __func__, __FILE__, __LINE__, ##__VA_ARGS__);        \
    }                                                                              \
    while (0)
#define st_print(S, ...)                                                           \
    do                                                                             \
    {                                                                              \
        fprintf(stdout, "%s @ %s:%d " S "\n",                                      \
                __extension__ __func__, __FILE__, __LINE__, ##__VA_ARGS__);        \
    }                                                                              \
    while (0)
#else
#define st_debug(S, ...)
#define st_mag(S, ...)
#define st_print(S, ...)
#endif


#ifdef ST_INFO
#define st_info(S, ...)                                                            \
    do                                                                             \
    {                                                                              \
        fprintf(stdout, KBLU "[INFO] %s @ %s:%d " KNRM S "\n",                     \
                __extension__ __func__, __FILE__, __LINE__, ##__VA_ARGS__);        \
    }                                                                              \
    while (0)
#define st_blu(S, ...)                                                             \
    do                                                                             \
    {                                                                              \
        fprintf(stdout, KBLU "%s @ %s:%d " KNRM S "\n",                            \
                __extension__ __func__, __FILE__, __LINE__, ##__VA_ARGS__);        \
    }                                                                              \
    while (0)
#else
#define st_info(S, ...)
#define st_blu(S, ...)
#endif


#ifdef ST_WARN
#define st_warn(S, ...)                                                            \
    do                                                                             \
    {                                                                              \
        fprintf(stdout, KYEL "[WARN] %s @ %s:%d " KNRM S "\n",                     \
                __extension__ __func__, __FILE__, __LINE__, ##__VA_ARGS__);        \
    }                                                                              \
    while (0)
#define st_yel(S, ...)                                                             \
    do                                                                             \
    {                                                                              \
        fprintf(stdout, KYEL "%s @ %s:%d " KNRM S "\n",                            \
                __extension__ __func__, __FILE__, __LINE__, ##__VA_ARGS__);        \
    }                                                                              \
    while (0)
#else
#define st_warn(S, ...)
#define st_yel(S, ...)
#endif


#ifdef ST_SUCCESS
#define st_success(S, ...)                                                         \
    do                                                                             \
    {                                                                              \
        fprintf(stdout, KGRN "[SUCCESS] %s @ %s:%d " KNRM S "\n",                  \
                __extension__ __func__, __FILE__, __LINE__, ##__VA_ARGS__);        \
    }                                                                              \
    while (0)
#define st_grn(S, ...)                                                             \
    do                                                                             \
    {                                                                              \
        fprintf(stdout, KGRN "%s @ %s:%d " KNRM S "\n",                            \
                __extension__ __func__, __FILE__, __LINE__, ##__VA_ARGS__);        \
    }                                                                              \
    while (0)
#else
#define st_success(S, ...)
#define st_grn(S, ...)
#endif


#ifdef ST_ERROR
#define st_error(S, ...)                                                           \
    do                                                                             \
    {                                                                              \
        fprintf(stdout, KRED "[ERROR] %s @ %s:%d " KNRM S "\n",                    \
                __extension__ __func__, __FILE__, __LINE__, ##__VA_ARGS__);        \
    }                                                                              \
    while (0)
#define st_red(S, ...)                                                             \
    do                                                                             \
    {                                                                              \
        fprintf(stdout, KRED "%s @ %s:%d " KNRM S "\n",                            \
                __extension__ __func__, __FILE__, __LINE__, ##__VA_ARGS__);        \
    }                                                                              \
    while (0)
#else
#define st_error(S, ...)
#define st_red(S, ...)
#endif


#endif  // ST_DEBUG_H
