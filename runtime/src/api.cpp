#include "common.h"

extern "C" {

bool fixSMForStream(void* stream, int minSM, int maxSM) {
    return fixSMForStream((cudaStream_t)stream, minSM, maxSM);
}

bool suggestSMForStream(void* stream, int minSM, int maxSM) {
    return suggestSMForStream((cudaStream_t)stream, minSM, maxSM);
}

}