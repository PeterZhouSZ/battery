#pragma once

#if defined(BATTERYLIB_EXPORT) 
#   define BLIB_EXPORT   __declspec(dllexport)
#else 
#   define BLIB_EXPORT   __declspec(dllimport)
#endif 