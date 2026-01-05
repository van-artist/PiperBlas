#pragma once
#include <stddef.h>
enum piState
{
    piSuccess = 0,
    piErrOpenFile = 1,
    piErrBadHeader = 2,
    piErrAlloc = 3,
    piErrIO = 4,
    piErrCSRInvalid = 5,
    piDataInvalid = 6,
    piNoMemory = 7,
    piFailure = -1
};
typedef enum piState piState;
