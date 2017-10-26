#pragma once
#include <functional>
#include <GL/glew.h>

#ifdef DEBUG

#ifdef _WIN32
#define THIS_FUNCTION __FUNCTION__
#else 
#define THIS_FUNCTION __PRETTY_FUNCTION__
#endif

#define S1(x) #x
#define S2(x) S1(x)
#define THIS_LINE __FILE__ " : " S2(__LINE__)

#define GL(x) x; GLError(THIS_LINE)

void logCerr(const char * label, const char * errtype);

bool GLError(
    const char *label = "",
    const std::function<void(const char *label, const char *errtype)>
        &callback = &logCerr);
#else
#define GL(x) x;
#define GLError(x) false
#endif


bool resetGL();