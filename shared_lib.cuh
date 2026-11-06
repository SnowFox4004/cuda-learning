#pragma once
#include "driver_types.h"
cudaError_t errorCheck(cudaError_t error_code, const char *filename,
                       int lineNo);
cudaError_t InitGPU();
