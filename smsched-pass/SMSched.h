#ifndef LLVM_TRANSFORMS_SMSCHED_H
#define LLVM_TRANSFORMS_SMSCHED_H

#include "llvm/IR/PassManager.h"

namespace llvm {
    class ForceInlinerPass : public PassInfoMixin<ForceInlinerPass> {
    public:
        PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
    };
    class SMSchedPass : public PassInfoMixin<SMSchedPass> {
    public:
        PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
    };
} // namespace llvm
#endif